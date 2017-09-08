/*
  Copyright (c) 2016-2017 M.I. Hollemans

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

import Foundation
import MetalPerformanceShaders

var imageOwners : [MPSImage : Set<Tensor>] = [:]
var writtenCounts : [MPSImage : Int] = [:]

extension MPSImage {
  var owners : Set<Tensor> {
    get {
      return imageOwners[self]!
    }
  }
  var writtenCount : Int {
    get {
      //print("Returning writtenCount for image \(self))")
      //print("WrittenCount for image \(self)) is \(writtenCounts[self]!) before read")
      return writtenCounts[self]!
    }
    set(count) {
      //print("Assigning writtenCount \(count) to image \(self))")
       writtenCounts[self] = count
      //print("WrittenCount for image \(self)) is \(writtenCounts[self]!) after write")
   }
  }
  func incrementWrittenCount() -> Void {
    writtenCounts[self] = writtenCounts[self]! + 1
  }
  func addOwner(tensor: Tensor) {
    if imageOwners[self] != nil {
      if imageOwners[self]!.contains(tensor) {
        fatalError("tensor \(tensor) already owns image \(self)")
      }
      imageOwners[self]!.insert(tensor)
    } else {
      imageOwners[self] = Set<Tensor>([tensor])
    }
  }
  func removeOwner(tensor: Tensor) {
    if imageOwners[self] != nil {
      if !imageOwners[self]!.contains(tensor) {
        fatalError("tensor (\tensor) does not ows image \(self)")
      }
      imageOwners[self]!.remove(tensor)
    } else {
      fatalError("tensor (\tensor) does not own any image)")
    }
  }
}

/**
  A tensor has a shape (width, height, depth). For each tensor there is an 
  `MPS(Temporary)Image` that holds its data.

  The model is really a DAG of tensors: Each tensor knows what layer it came 
  from and what the input tensor to that layer was. Each tensor also knows
  what other tensors depend on it.
*/
public class Tensor {
  /**
    Whether this tensor writes its results into an MPSTemporaryImage. Normally
    this is true for all tensor except the last. You can override this if you
    want to keep track of the tensor's MPSImage for processing afterwards.
  */
  public var imageIsTemporary = true
  //public var imageIsTemporary = false

  // The layer that takes the data from `input`, transforms it, and stores it
  // inside this tensor. Is nil for Input and Concatenate tensors.
  var layer: Layer?

  // Which tensor this one depends on. Together with `next`, this describes
  // the model's graph.
  var input: Tensor?

  // The tensors that depend on this one, i.e. which tensors we are the `input` 
  // for. The number of tensors in this array is also the readCount for this
  // tensor's MPSTemporaryImage.
  var next: [Tensor] = []
  var previous: [Tensor] = []

  // The shape of the tensor is determined by its input tensor and the layer.
  var shape = DataShape()

  // For debugging and printing the model summary.
  var typeName = "Tensor"
  var id = "" // can be used to identify tensors like Concatente and Collect that do not have a layer

  // Used to set offset and clipRect for reading from another tensor's image.
  //var sourceChannelOffset = 0

  // Used to set destinationFeatureChannelOffset for merging the output from
  // multiple layers into one image.
  var destinationChannelOffset = 0

  // Used to set destinationImage for collecting the output from
  // multiple layers into one image.
  var destinationImage = 0
    /*
    var destinationImage : Int {
        get {
            return _destinationImage
        }
        set(newDestination) {
            _destinationImage = newDestination
        }
    }
    public func totalChannelDestinationOffSet() -> Int {
        return ((shape.channels + 3) / 4) * 4 * destinationImage + destinationChannelOffset
    }
   */

  // If this is set, the layer will write into the MPSImage for the destination
  // tensor. If nil, a new (temporary) image is allocated for the tensor and we
  // write into that. Usually this will be nil.
  var _destinationTensor: Tensor?
  
  var destinationTensor : Tensor? {
    get {
      return _destinationTensor
    }
    set(newTensor) {
      _destinationTensor = newTensor
    }
  }
  var shortId : String {
    get {
      let sid = layer?.name ?? "**\(typeName)-\(id)**"
      return sid
    }
  }
  
  // The image that the layer for this tensor will write into. Since the layers
  // may not be processed in their original order (depending on the topological
  // sort), this is how we keep track of which MPSImage to use where. Note that
  // image may point to the destinationTensor's image.
  var _image: MPSImage?
  internal(set) public var image : MPSImage? {
        get {
            return _image
        }
        set(newImage) {
          if _image != nil {
            let layerName = layer?.name ?? shortId
            print("Remove output image for Tensor:" + layerName + ", owners = \(_image!.owners)")
            _image!.removeOwner(tensor: self)
          }
          if newImage != nil {
            newImage!.addOwner(tensor: self)
            let layerName = layer?.name ?? shortId
            print("Set output image for Tensor:" + layerName + ", owners = \(newImage!.owners)")
          }
          _image = newImage
          //print(self," - image set to", String(describing: _image))
         }
    }
  // Reference count. It is used to set the readCount of the MPSTemporyImage
  // for this tensor, but also tells us when to set the `image` property to nil
  // when we're done using it (so that we don't hang on to images for longer
  // than necessary).
  var readCount = 1
  var releasedReadCount : String?
  
  // Tensor with mutiple layers writing into maintain a written count, too
  // so they won't be discarded too early until all the inbound layers have
  // writtenn into the underlying image
  // managed
  var writtenCount = 0

  fileprivate init() { }

  /**
    Connects the `input` tensor to the `layer` and creates the output tensor
    that holds the results of the layer's computations.
    
    The shorthand way of writing this is:

        let output = input --> layer

    which just does:

        let output = Tensor(input: input, layer: layer)
  */
  public init(input: Tensor, layer: Layer) {
    self.input = input
    self.layer = layer

    input.next.append(self)
    self.previous.append(input)
    shape = layer.outputShape(for: input.shape)
    print("creating new shape of input:", input, "to layer:", layer,"shape:",shape)
  }

  func releasePrevious() {
    for prev in previous {
      
    }
  }
  
  func release(byLayer: Layer) {
    print("Decrementing readcount for input tensor \(self.shortId) of layer \(byLayer.name), current readCount = \(self.readCount)")
    self.readCount -= 1
    if let image = self.image as? MPSTemporaryImage {
      print("Image for input tensor \(self.shortId) of layer \(byLayer.name), has readCount = \(image.readCount)")
      if (image.readCount > 0) {
        print("decrementing readcount for input image by its writtenCount = \(image.writtenCount)")
        image.readCount -= image.writtenCount
      }
    }
    
    if self.readCount <= 0 {
      print("Deleting image of input tensor \(self.shortId) of layer \(byLayer.name)")
      
      if let image = self.image as? MPSTemporaryImage {
        self.image = nil
        if image.owners.count == 0 {
          self.releasedReadCount = image.readCount.description
        } else {
          self.releasedReadCount = image.readCount.description + "*"
        }
        print("Image for input tensor \(self.shortId) of layer \(byLayer.name), released with readCount = \(image.readCount), owners=\(image.owners.description)")
      } else {
        self.image = nil
      }
    }
  }
  
  func summary() -> String {
    let layerName = layer?.name ?? shortId
    let layerType = layer?.typeName ?? "Tensor"
    let paramCount = layer?.paramCount ?? 0
    let outputs = self.next.count
    let inputs = self.previous.count

    let n = layerName.padding(toLength: 30, withPad: " ", startingAt: 0)
    let t = layerType.padding(toLength: 10, withPad: " ", startingAt: 0)
    let o = shape.debugDescription.padding(toLength: 16, withPad: " ", startingAt: 0)
    let p = "\(paramCount)".padding(toLength: 13, withPad: " ", startingAt: 0)
    let ios = "\(inputs)/\(outputs)".padding(toLength: 10, withPad: " ", startingAt: 0)

    let s = String(format: "%@ %@ %@ %@ %@", n, t, o, p, ios)
    //      + "\(destinationChannelOffset)"
    return s
  }
  func debugSummary(isCurrent: Bool, marker: String) -> String {
    let layerName = layer?.name ?? shortId
    let layerType = layer?.typeName ?? "Tensor"
    let paramCount = layer?.paramCount ?? 0
    let outputs = self.next.count
    let inputs = self.previous.count
    let imageDescr = "\(self.image?.description ?? "nil")".padding(toLength: 32, withPad: " ", startingAt: 0)

    let n = layerName.padding(toLength: 30, withPad: " ", startingAt: 0)
    let t = layerType.padding(toLength: 10, withPad: " ", startingAt: 0)
    let o = shape.debugDescription.padding(toLength: 16, withPad: " ", startingAt: 0)
    let p = "\(paramCount)".padding(toLength: 13, withPad: " ", startingAt: 0)
    let ios = "\(inputs)/\(outputs)".padding(toLength: 6, withPad: " ", startingAt: 0)
    
    let writtenCount = "\(self.image?.writtenCount.description ?? "-")"
    let tensorReadCount = "\(self.readCount)".padding(toLength: 3, withPad: " ", startingAt: 0)

    let releaseReadCount = "\(self.releasedReadCount?.description ?? "-")"

    var readCount = "-"
    if let tempImage = self.image as? MPSTemporaryImage {
      readCount = "\(tempImage.readCount)"
    }
    let indicator = isCurrent ? marker : ""
    let s = String(format: "%@ %@ %@ %@ %@ %@ %@ %@ %@ %@ %@", n, t, o, p, tensorReadCount, ios, imageDescr, writtenCount, readCount, releaseReadCount, indicator)
    //      + "\(destinationChannelOffset)"
    return s
  }
}



extension Tensor: Hashable {
  // Needs to be hashable because for tensors whose imageIsTemporary flag is
  // false, we use a dictionary to find the corresponding MPSImage objects.
  // Since each tensor is a unique entity, we use the tensor's address as the
  // hash value (this is similar to how NSObjects are hashed).
  public var hashValue: Int {
    return unsafeBitCast(self, to: Int.self)
  }
}

public func == (lhs: Tensor, rhs: Tensor) -> Bool {
  return lhs === rhs
}

extension Tensor: CustomDebugStringConvertible {
  public var debugDescription: String {
    return "Tensor, shape \(shape), layer " + (layer?.name ?? typeName)
  }
}

/**
  A placeholder for input. Your model always starts with an Input tensor.
  
  You can leave the shape of this tensor completely or partially unspecified.
  However, if you do specify a size, this is used to force the input texture 
  to be in a specific shape.
  
  If your first layer is `Resize`, which takes a texture of arbitrary size and
  scales it to a fixed size, then you can specify `Input()` without a shape.
  
  However, if your first layer is something like a `Convolution`, then you need
  `Input` to specify the size of the texture that goes into the conv layer. 
  (Without it, we won't know how large the `Convolution` layer's output will be
  and as a result we can't allocate an MPSTemporaryImage for it.)
*/
public func Input(width: Int? = nil, height: Int? = nil, channels: Int? = nil, numImages: Int = 1) -> Tensor {
  let tensor = Tensor()
  tensor.typeName = "Input"
  tensor.shape = DataShape(width: width ?? -1,
                           height: height ?? -1,
                           channels: channels ?? -1,
                           numImages: numImages)
  return tensor
}

/**
  Depth-concatenates several tensors into one large tensor.
*/
public func Concatenate(_ tensors: [Tensor], name:String = "") -> Tensor {
  let merged = Tensor()

  var maxWidth = 0
  var maxHeight = 0
  var channels = 0

  for input in tensors {
    // Tell the other tensor that it should write into our image and not
    // an image of its own.
    input.destinationChannelOffset = channels
    input.destinationTensor = merged

    // Figure out how large to make the merged tensor's destination image.
    maxWidth = max(maxWidth, input.shape.width)
    maxHeight = max(maxHeight, input.shape.height)
    channels += input.shape.channels

    // Connect each tensor to the merged tensor, or the topological sort
    // will fail and the graph will be incomplete.
    input.next.append(merged)
    merged.previous.append(input)
  }

  merged.shape = DataShape(width: maxWidth, height: maxHeight, channels: channels)
  merged.typeName = "Concat"
  merged.id = name

  // Note: We don't fill in the `input` property because we potentially have
  // multiple inputs, not just one. This is no problem because Concatenate is
  // skipped during encoding (as it has no layer).

  return merged
}

/**
 Collects several tensors into images of one tensor.
 */
public func Collect(_ tensors: [Tensor], name:String = "") -> Tensor  {
  print("Creating Collection Tensor of:",tensors)
  let collected = Tensor()
  
  var image = 0
    
  // the first tensors dimensions define the dimensions the other tensors have to match
  let channels = tensors[0].shape.channels
  let width = tensors[0].shape.width
  let height = tensors[0].shape.height
  for input in tensors {
    //precondition(input.shape.channels != channels, "Number of channels must be equal in merge channel")
    print("channels:",channels, "input.shape.channels:",input.shape.channels)
    
    // Tell the other tensor that it should write into our image and not
    // an image of its own.
    input.destinationImage = image
    input.destinationTensor = collected
    
    // Figure out how large to make the collecting tensor's destination image.
    if input.shape.width != width {
      fatalError("input tensor #"+String(image)+" of Collect function does not match width of first input tensor")
    }
    if input.shape.height != height {
      fatalError("input tensor #"+String(image)+" of Collect function does not match height of first input tensor")
    }

    image += input.shape.numImages
    
    // Connect each tensor to the collecting tensor, or the topological sort
    // will fail and the graph will be incomplete.
    input.next.append(collected)
    collected.previous.append(input)
  }
  
  collected.shape = DataShape(width: width, height: height, channels: channels, numImages:image)
  collected.typeName = "Collect"
  collected.id = name
  print("Collected Shape:",collected.shape)

  // Note: We don't fill in the `input` property because we potentially have
  // multiple inputs, not just one. This is no problem because Concatenate is
  // skipped during encoding (as it has no layer).
  
  return collected
}
