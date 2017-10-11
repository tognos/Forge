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
import Metal
import MetalPerformanceShaders

public enum PaddingType {
  case same    // add zero padding
  case valid   // don't add padding
}

func offsetForConvolution(padding: PaddingType,
                          sourceWidth: Int,
                          sourceHeight: Int,
                          destinationWidth: Int,
                          destinationHeight: Int,
                          kernelWidth: Int,
                          kernelHeight: Int,
                          strideInPixelsX: Int,
                          strideInPixelsY: Int) -> MPSOffset {
  if padding == .same {
    let padH = (destinationHeight - 1) * strideInPixelsY + kernelHeight - sourceHeight
    let padW = (destinationWidth  - 1) * strideInPixelsX + kernelWidth  - sourceWidth
    /*
    let testDestinationHeight = (sourceHeight - kernelHeight + padH) / strideInPixelsX + 1
    let testDestinationWidth = (sourceWidth - kernelWidth + padW) / strideInPixelsY + 1
    assert(destinationHeight == testDestinationHeight)
    assert(destinationWidth == testDestinationWidth)
    */
    // TODO: above offset calculation seems ok at first glance, but yields negative
    // pad values for kernel size 1 and stride 2, so we just clamp it to zero for now,
    // but the calculation should be checked for other possible failure cases
    let padWC = max(padW, 0)
    let padHC = max(padH, 0)

    //let offset = MPSOffset(x: (kernelWidth - padW)/2, y: (kernelHeight - padH)/2, z: 0)
    let offset = MPSOffset(x: (kernelWidth - padWC)/2, y: (kernelHeight - padHC)/2, z: 0)
    /*
    if strideInPixelsX != 1 {
      print("offsetForConvolution same: padH: \(padH), padW: \(padW)" )
      print("offsetForConvolution same: offset: \(offset)" )
    }
    */
    return offset
  } else {
    let offset = MPSOffset(x: kernelWidth/2, y: kernelHeight/2, z: 0)
    //let offset = MPSOffset(x: 0, y: 0, z: 0)
    //print("offsetForConvolution valid: offset: \(offset)" )
    return offset
  }
}

func offsetForPooling(padding: PaddingType,
                      sourceWidth: Int,
                      sourceHeight: Int,
                      kernelWidth: Int,
                      kernelHeight: Int,
                      strideInPixelsX: Int,
                      strideInPixelsY: Int) -> MPSOffset {
  if padding == .same {
    var offset = MPSOffset(x: 0, y: 0, z: 0)
    if kernelWidth % 2 == 0 {
      offset.x += (((sourceWidth - 1) % strideInPixelsX) / 2) + 1
    } else {
      offset.x += (((sourceWidth - 1) % strideInPixelsX) + 1) / 2
    }
    if kernelHeight % 2 == 0 {
      offset.y += (((sourceHeight - 1) % strideInPixelsY) / 2) + 1
    } else {
      offset.y += (((sourceHeight - 1) % strideInPixelsY) + 1) / 2
    }
    return offset
  } else {
    return MPSOffset(x: kernelWidth/2, y: kernelHeight/2, z: 0)
  }
}

/**
  The abstract base class for all layers. You should not create instances of
  this class directly.
*/
open class Layer {
  internal(set) public var name: String

  // Most layers take MPSImages as input but for Resize it's more optimal to
  // work directly on the input texture. That saves making an MPSImage object.
  // Probably a premature optimization. ;-)
  internal(set) public var wantsTextures: Bool

  // Most layers require that the complete shape of the input tensor is known.
  // However, some layers (such as Resize and Custom) can handle inputs of any
  // size. If your first layer is a type that must know the size (Convolution)
  // then you need to specify that size to the Input tensor.
  internal(set) public var allowsIncompleteShape: Bool

  /* Whether this layer uses bias terms in addition to weights. */
  internal(set) public var useBias: Bool

  // The same layer can be used by multiple tensors, but we should only create
  // its compute just once. Reusing layers is mostly useful for things like
  // pooling, which don't take parameters.
  var createdCompute = false

  // The parameter count shown in the summary. (Filled in by the compiler.)
  var paramCount = 0

  public init(name: String = "",
              useBias: Bool = true,
              wantsTextures: Bool = false,
              allowsIncompleteShape: Bool = false) {
    self.name = name
    self.useBias = useBias
    self.wantsTextures = wantsTextures
    self.allowsIncompleteShape = allowsIncompleteShape
  }

  /* Subclasses must implement these methods. */

  open var typeName: String {
    fatalError("Subclass must implement this function")
  }

  open func outputShape(for inputShape: DataShape) -> DataShape {
    fatalError("Subclass must implement this function")
  }

  open func createCompute(device: MTLDevice,
                          inputShape: DataShape,
                          outputShape: DataShape,
                          weights: ParameterData?,
                          biases: ParameterData?) throws {
    // do nothing
  }

  open func encode(commandBuffer: MTLCommandBuffer,
                   sourceTensor: Tensor,
                   destinationTensor: Tensor) {
    // Note: sourceTensor.image and destinationTensor.image are guaranteed
    // to be non-nil at this point, so it's OK to force-unwrap them.
  }

  open func encode(commandBuffer: MTLCommandBuffer,
                   sourceTensor: Tensor,
                   sourceTexture: MTLTexture,
                   destinationTensor: Tensor) {
    // This is a special-case method for layers that prefer to work with
    // textures rather than MPSImages. The output will always be a texture
    // from an MPSImage but this not necessarily true for the input texture.
  }

  open func weightCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return 0
  }

  open func biasCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return 0
  }
}

extension Layer: CustomDebugStringConvertible {
  public var debugDescription: String {
    return name
  }
}

/**
  Abstract base class for layers that encode a single MPSCNN kernel.
*/
public class MPSCNNLayer: Layer {
  var mpscnn: MPSCNNKernel!
  var encodedOffset: MPSOffset!

  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {

    // FUTURE: For a residual connection, where we may want to read from the
    // destination image (and write to that same destination image), we would
    // set mpscnn.offset and clipRect here using sourceTensor's channel offset.

    /*
    if true {
        print("MPSCNNLayer: Encoding for layer:", self)
        print("MPSCNNLayer:sourceTensor:",sourceTensor,", destinationTensor:",destinationTensor)
      if let image = sourceTensor.image as? MPSTemporaryImage {
        print("MPSCNNLayer: sourceTensor image:", image.debugDescription)
      }
      if let image = destinationTensor.image as? MPSTemporaryImage {
        print("MPSCNNLayer: destinationTensor image:",image.debugDescription)
      }
    } else {
        print("MPSCNNLayer: Encoding for layer:", self)
    }
    */
    
    mpscnn.destinationFeatureChannelOffset = destinationTensor.destinationChannelOffset
    //print("mpscnn.destinationFeatureChannelOffset:",mpscnn.destinationFeatureChannelOffset)

    mpscnn.offset = encodedOffset
    mpscnn.offset.z = sourceTensor.destinationImageNumber
    
    //mpscnn.offset = MPSOffset(x: 0, y: 0, z: 0)
    mpscnn.clipRect.origin.z = destinationTensor.destinationImageNumber
    mpscnn.clipRect.size.depth = 1
    //print("MPSCNNLayer: Encoding for layer '\(self.name), sourceTensor='\(sourceTensor.shortId)', src offset=\(mpscnn.offset), src image=\(sourceTensor.image?.description ?? "nil")")
    //print("cliprect:", mpscnn.debugDescription)
    
//    if let image = sourceTensor.image as? MPSTemporaryImage {
//        print("Before encode: sourceImage.readCount = \(image.readCount)")
//    }
    mpscnn.encode(commandBuffer: commandBuffer,
                  sourceImage: sourceTensor.image!,
                  destinationImage: destinationTensor.image!)
    destinationTensor.written(byLayer: self)
//    if let image = sourceTensor.image as? MPSTemporaryImage {
//        print("After encode: sourceImage.readCount = \(image.readCount)")
//    }
  }
}

public class SpaceToDepthX2: Layer {
    
    /**
     Creates a new layer that reshapes input to half spatial size of four times the number of filters
     The current implementation is limited to input that has at least 4 layers because you can't blit
     between textures of different pixel format; to efficiently convert between different pixel formats
     would require a custom compute shader.
     Another restriction is that numImages must be 1, but this is currently a restriction for most Forge
     layers anyway.
     */
    
    public init(name: String = "") {
        super.init(name: name)
    }
    
    override public var typeName: String {
        return "SpaceToDepthX2"
    }
    
    override public func outputShape(for inputShape: DataShape) -> DataShape {
        return DataShape(width: inputShape.width/2,
                         height: inputShape.height/2,
                         channels: inputShape.channels*4,
                         numImages: inputShape.numImages
        )
    }
    
    override public func encode(commandBuffer: MTLCommandBuffer,
                                sourceTensor: Tensor,
                                destinationTensor: Tensor) {
        let sourceImage = sourceTensor.image!
        let destImage = destinationTensor.image!
        
        if sourceImage.featureChannels < 4 {
            fatalError("We are sorry, the current implementation of SpaceToDepthX2 requires the input to have at least 4 channels")
        }
        if sourceImage.numberOfImages > 1 {
            fatalError("We are sorry, the current implementation of SpaceToDepthX2 requires the input to have only 1 image")
        }
        let width = destinationTensor.shape.width
        let height = destinationTensor.shape.height
        let sourceRegion = [MTLRegionMake3D(0,     0,      0, width, height, 1),
                            MTLRegionMake3D(width, 0,      0, width, height, 1),
                            MTLRegionMake3D(0,     height, 0, width, height, 1),
                            MTLRegionMake3D(width, height, 0, width, height, 1)]
        let destOrigin = MTLOrigin(x: 0, y: 0, z: destinationTensor.destinationImageNumber)
        
        let srcChannels = sourceTensor.shape.channels
        let srcSlices = (srcChannels + 3)/4
        
        if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
            
            for slice in 0..<srcSlices {
                for region in 0...3 {
                blitEncoder.copy(from: sourceImage.texture,
                                 sourceSlice: slice,
                                 sourceLevel: 0,
                                 sourceOrigin: sourceRegion[region].origin,
                                 sourceSize: sourceRegion[region].size,
                                 to: destImage.texture,
                                 destinationSlice: slice * 4 + region,
                                 destinationLevel: 0,
                                 destinationOrigin: destOrigin)
                }
            }
            
            blitEncoder.endEncoding()
        } else {
            print("Error: SpaceToDepthX2.SpaceToDepthX2: cant create blitEncoder")
        }
        if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
        }
        destinationTensor.written(byLayer: self)
    }
}

public class ZeroPadding: Layer {
  
  /**
   Creates a new layer that creates an output enlarged by outer zero padding
   */
  
  let topPadding: Int
  let bottomPadding: Int
  let leftPadding: Int
  let rightPadding: Int
  let padValue: Float
  
  var padKernel: PadKernel? = nil

  public init(tblr_padding: (Int, Int, Int, Int),
              padValue: Float = 0,
              name: String = "") {
    self.topPadding = tblr_padding.0
    self.bottomPadding = tblr_padding.1
    self.leftPadding = tblr_padding.2
    self.rightPadding = tblr_padding.3
    self.padValue = padValue
    super.init(name: name)
  }
  
  override public var typeName: String {
    return "ZeroPadding"
  }
  
  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: inputShape.width + leftPadding + rightPadding,
                     height: inputShape.height + topPadding + bottomPadding,
                     channels: inputShape.channels)
  }
  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {
    self.padKernel = PadKernel(device: device,
                               tblr_padding: (topPadding, bottomPadding, leftPadding, rightPadding),
                               padValue: self.padValue,
                               featureChannels: inputShape.channels,
                               writesToArray: outputShape.channels > 4 || outputShape.numImages > 1)
  }
  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {
    precondition(destinationTensor.destinationChannelOffset % 4 == 0, "destinationChannelOffset must be a multiple of 4")
    let destinationSliceOffset = (destinationTensor.destinationChannelOffset / 4) +
      destinationTensor.destinationImageNumber * (sourceTensor.shape.channels+3)/4
    self.padKernel?.encode(commandBuffer: commandBuffer,
                           sourceImage: sourceTensor.image!,
                           destinationImage: destinationTensor.image!,
                           destinationSliceOffset: destinationSliceOffset)
    /*
    if let image = sourceTensor.image! as? MPSTemporaryImage {
      image.readCount -= 1
    }
     */
    destinationTensor.written(byLayer: self)
    
  }

/*
   // This version using the blitencoder works, but depending on what happens before
   // it exhibits strange behavior like rescaling all values to a range from 0..1,
   // returning an all black image or strange data; therefore we use a
   // above custom kernel for the procedure
  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {
    let sourceImage = sourceTensor.image!
    let destImage = destinationTensor.image!
    
    let width = sourceTensor.shape.width
    let height = sourceTensor.shape.height
    let sourceRegion = MTLRegionMake3D(0,     0,      0,
                                       width, height, 1)
    
    let destOrigin = MTLOrigin(x: leftPadding, y: topPadding, z: destinationTensor.destinationImage)
    
    let srcChannels = sourceTensor.shape.channels
    let srcSlices = (srcChannels + 3)/4
    
    if let blitEncoder = commandBuffer.makeBlitCommandEncoder() {
      
      let buffer = destImage.texture.buffer!
      blitEncoder.__fill(buffer, range: NSMakeRange(0, buffer.length), value: 0)
      
      for slice in 0..<srcSlices {
          print("ZeroPadding:encode: blitEncoder copying slice \(slice)")
          blitEncoder.copy(from: sourceImage.texture,
                           sourceSlice: slice,
                           sourceLevel: 0,
                           sourceOrigin: sourceRegion.origin,
                           sourceSize: sourceRegion.size,
                           to: destImage.texture,
                           destinationSlice: slice,
                           destinationLevel: 0,
                           destinationOrigin: destOrigin)
        
      }
      
      print("ZeroPadding:encode: sourceImage=\(sourceImage.debugDescription)\nsourceImage.texture=\(sourceImage.texture.debugDescription)")
      print("ZeroPadding:encode: destImage=\(destImage.debugDescription)\ndestImage.texture=\(destImage.texture.debugDescription)")
      blitEncoder.endEncoding()
      //blitEncoder.synchronize(resource: destImage.texture)
    } else {
      print("Error: ZeroPadding2D.encode: cant create blitEncoder")
    }
    if let image = sourceImage as? MPSTemporaryImage {
      image.readCount -= 1
    }
    destinationTensor.written()
  }
 */
}

public class MergeOperation: Layer {
  
  var mopKernel: MergeOpKernel?
  var operation: MergeOpType
    
  /**
   Creates a new layer that sums up all the images in the tensor
   */
  
  public init(operation: MergeOpType, name: String = "") {
    self.operation = operation
    super.init(name: name)
  }
  
  override public var typeName: String {
    return "\(operation)"
  }
  
  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: inputShape.width,
                     height: inputShape.height,
                     channels: inputShape.channels,
                     numImages: 1)
  }
  
  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {
    
    mopKernel!.encode(commandBuffer: commandBuffer,
               sourceImage: sourceTensor.image!,
               destinationImage: destinationTensor.image!,
               destinationChannelOffset: destinationTensor.destinationChannelOffset,
               destinationImageNumber: destinationTensor.destinationImageNumber)
    destinationTensor.written(byLayer: self)
  }
    override public func createCompute(device: MTLDevice,
                                       inputShape: DataShape,
                                       outputShape: DataShape,
                                       weights: ParameterData?,
                                       biases: ParameterData?) throws {
        mopKernel = MergeOpKernel(device: device,
                                  inputFeatureImages: inputShape.numImages,
                                  featureChannels: inputShape.channels,
                                  featureOp: self.operation)
    }
}

//   case Add = 1, Multiply, Maximum, Average

public class Add: MergeOperation {
  public init(name: String = "") {
    super.init(operation: .Add, name: name)
  }
}
public class Multiply: MergeOperation {
  public init(name: String = "") {
    super.init(operation: .Multiply, name: name)
  }
}
public class Maximum: MergeOperation {
  public init(name: String = "") {
    super.init(operation: .Maximum, name: name)
  }
}
public class Average: MergeOperation {
  public init(name: String = "") {
    super.init(operation: .Average, name: name)
  }
}

/**
  Convolutional layer.
*/
public class Convolution: MPSCNNLayer {
  let kernel: (Int, Int)
  let channels: Int
  let stride: (Int, Int)
  let padding: PaddingType
  let activation: MPSCNNNeuron?
  var conv: MPSCNNConvolution!

  /**
    Creates a convolution layer.
  
    - Parameters:
      - kernel: `(width, height)`
      - channels: Number of output channels.
      - stride: `(x, y)`
      - padding: If .same, the output width and height are the same as the
        input width and height. (This uses zero padding.)
      - useBias: whether this layer uses bias terms in addition to the weights
      - name: The name is used to load the layer's parameters.
  */
  public init(kernel: (Int, Int),
              channels: Int,
              stride: (Int, Int) = (1, 1),
              padding: PaddingType = .same,
              activation: MPSCNNNeuron? = nil,
              useBias: Bool = true,
              name: String = "") {
    self.kernel = kernel
    self.channels = channels
    self.stride = stride
    self.padding = padding
    self.activation = activation
    super.init(name: name, useBias: useBias)
  }

  override public var typeName: String {
    return "Conv"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    if padding == .same {
      return DataShape(width: (inputShape.width - 1)  / stride.0 + 1,
                      height: (inputShape.height - 1) / stride.1 + 1,
                    channels: channels)
    } else {
      return DataShape(width: (inputShape.width  - kernel.0) / stride.0 + 1,
                      height: (inputShape.height - kernel.1) / stride.1 + 1,
                    channels:  channels)
    }
  }

  override public func weightCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return inputShape.channels * kernel.1 * kernel.0 * outputShape.channels
  }

  override public func biasCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return useBias ? outputShape.channels : 0
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {

    guard let weights = weights else {
      throw ModelError.compileError(message: "missing weights for layer '\(name)'")
    }
    if useBias && biases == nil {
      throw ModelError.compileError(message: "missing bias terms for layer '\(name)'")
    }

    let desc = MPSCNNConvolutionDescriptor(kernelWidth: kernel.0,
                                           kernelHeight: kernel.1,
                                           inputFeatureChannels: inputShape.channels,
                                           outputFeatureChannels: outputShape.channels,
                                           neuronFilter: activation)
    desc.strideInPixelsX = stride.0
    desc.strideInPixelsY = stride.1

    conv = MPSCNNConvolution(device: device,
                             convolutionDescriptor: desc,
                             kernelWeights: weights.pointer,
                             biasTerms: biases?.pointer,
                          flags: .none)
    conv.edgeMode = .zero
    /*
    print("Neuron=\(conv.neuronType.rawValue)")
    print("NeuronA=\(conv.neuronParameterA)")
    print("NeuronB=\(conv.neuronParameterB)")
    */
    mpscnn = conv
  }

  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {

    // We compute the padding at encode-time, so that this layer can be
    // reused on tensors of different sizes. Note that the input and output
    // depth must not vary, only the width and height may be different.
    /*
    if stride.0 != 1 {
      print("Ecoding for Layer \(self.name)")
    }
    */
    self.encodedOffset = offsetForConvolution(padding: padding,
                                              sourceWidth: sourceTensor.shape.width,
                                              sourceHeight: sourceTensor.shape.height,
                                              destinationWidth: destinationTensor.shape.width,
                                              destinationHeight: destinationTensor.shape.height,
                                              kernelWidth: kernel.0,
                                              kernelHeight: kernel.1,
                                              strideInPixelsX: stride.0,
                                              strideInPixelsY: stride.1)

    super.encode(commandBuffer: commandBuffer,
                 sourceTensor: sourceTensor,
                 destinationTensor: destinationTensor)
  }
}

/**
  Abstract base class for max-pooling and average-pooling layers.
*/
public class Pooling: MPSCNNLayer {
  let kernel: (Int, Int)
  let stride: (Int, Int)
  let padding: PaddingType
  let edgeMode: MPSImageEdgeMode
  var pool: MPSCNNPooling!

  /**
    Creates a new pooling layer.
    
    - Parameters:
      - kernel: `(width, height)`
      - stride: `(x, y)`
      - padding: Whether to add padding around the input image. (This uses 
                 "clamp" padding.)
  */
  public init(kernel: (Int, Int),
              stride: (Int, Int),
              padding: PaddingType = .valid,
              edgeMode: MPSImageEdgeMode = .clamp,
              name: String = "") {
    self.kernel = kernel
    self.stride = stride
    self.padding = padding
    self.edgeMode = edgeMode
    super.init(name: name)
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    if padding == .same {
      return DataShape(width: (inputShape.width - 1)  / stride.0 + 1,
                      height: (inputShape.height - 1) / stride.1 + 1,
                    channels: inputShape.channels)
    } else {
      return DataShape(width: (inputShape.width  - kernel.0) / stride.0 + 1,
                      height: (inputShape.height - kernel.1) / stride.1 + 1,
                    channels:  inputShape.channels)
    }
  }

  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {

    // We compute the padding at encode-time, so that this layer can be
    // reused on tensors of different sizes.
    self.encodedOffset = offsetForPooling(padding: padding,
                                   sourceWidth: sourceTensor.shape.width,
                                   sourceHeight: sourceTensor.shape.height,
                                   kernelWidth: kernel.0,
                                   kernelHeight: kernel.1,
                                   strideInPixelsX: stride.0,
                                   strideInPixelsY: stride.1)

    super.encode(commandBuffer: commandBuffer,
                 sourceTensor: sourceTensor,
                 destinationTensor: destinationTensor)
  }
}

/**
  Max-pooling layer.
*/
public class MaxPooling: Pooling {
  override public var typeName: String {
    return "MaxPool"
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {

    pool = MPSCNNPoolingMax(device: device,
                            kernelWidth: kernel.0,
                            kernelHeight: kernel.1,
                            strideInPixelsX: stride.0,
                            strideInPixelsY: stride.1)
    pool.edgeMode = edgeMode
    mpscnn = pool
  }
}

/**
  Average-pooling layer.
*/
public class AveragePooling: Pooling {
  override public var typeName: String {
    return "AvgPool"
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {

    pool = MPSCNNPoolingAverage(device: device,
                                kernelWidth: kernel.0,
                                kernelHeight: kernel.1,
                                strideInPixelsX: stride.0,
                                strideInPixelsY: stride.1)
    pool.edgeMode = edgeMode
    mpscnn = pool
  }
}

/**
  Global average-pooling layer
  
  This does the same thing as an AveragePooling layer with a kernel size equal
  to the input's spatial dimensions. If the input image is WxHxC, this averages
  across the width and height, and outputs a 1x1xC image.
*/
public class GlobalAveragePooling: MPSCNNLayer {
  override public var typeName: String {
    return "GlbAvgPool"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: 1, height: 1, channels: inputShape.channels)
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {

    let pool = MPSCNNPoolingAverage(device: device,
                                    kernelWidth: inputShape.width,
                                    kernelHeight: inputShape.height,
                                    strideInPixelsX: inputShape.width,
                                    strideInPixelsY: inputShape.height)

    self.encodedOffset = MPSOffset(x: inputShape.width/2, y: inputShape.height/2, z: 0)
    pool.edgeMode = .clamp
    self.mpscnn = pool
  }
}

/**
  Fully-connected layer.
*/
public class Dense: MPSCNNLayer {
  let neurons: Int
  let activation: MPSCNNNeuron?

  /**
    Creates a fully-connected layer.
  
    - Parameters:
      - neurons: The number of neurons in this layer.
      - name: The name is used to load the layer's parameters.
  */
  public init(neurons: Int,
              activation: MPSCNNNeuron? = nil,
              useBias: Bool = true,
              name: String = "") {
    self.neurons = neurons
    self.activation = activation
    super.init(name: name, useBias: useBias)
    self.encodedOffset = MPSOffset(x: 0, y: 0, z: 0)
  }

  override public var typeName: String {
    return "Dense"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: 1, height: 1, channels: neurons)
  }

  override public func weightCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return inputShape.channels * inputShape.height * inputShape.width * neurons
  }

  override public func biasCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return useBias ? neurons : 0
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {

    guard let weights = weights else {
      throw ModelError.compileError(message: "missing weights for layer '\(name)'")
    }

    // A fully-connected layer is a special version of a convolutional layer
    // where the kernel size is equal to the width/height of the input volume.
    // The output volume is 1x1xfanOut.
    let desc = MPSCNNConvolutionDescriptor(kernelWidth: inputShape.width,
                                           kernelHeight: inputShape.height,
                                           inputFeatureChannels: inputShape.channels,
                                           outputFeatureChannels: neurons,
                                           neuronFilter: activation)

    // NOTE: For some reason MPSCNNFullyConnected crashes when we write
    // biases?.pointer, which makes no sense at all since it works fine
    // for MPSCNNConvolution.
    var biasTerms: UnsafeMutablePointer<Float>?
    if useBias {
      guard let biases = biases else {
        throw ModelError.compileError(message: "missing bias terms for layer '\(name)'")
      }
      biasTerms = biases.pointer
    }

    mpscnn = MPSCNNFullyConnected(device: device,
                                  convolutionDescriptor: desc,
                                  kernelWeights: weights.pointer,
                                  biasTerms: biasTerms,
                                  flags: .none)
  }
}

/**
  Softmax layer.
*/
public class Softmax: MPSCNNLayer {
  override public var typeName: String {
    return "Softmax"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return inputShape
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {
    self.encodedOffset = MPSOffset(x: 0, y: 0, z: 0)
    mpscnn = MPSCNNSoftMax(device: device)
  }
}

/**
  Lets you use any MPSCNNNeuron as a layer of its own.
*/
public class Activation: MPSCNNLayer {
  public init(_ activation: MPSCNNNeuron, name: String = "") {
    super.init(name: name)
    self.mpscnn = activation
    self.encodedOffset = MPSOffset(x: 0, y: 0, z: 0)
  }

  override public var typeName: String {
    return "Activation"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return inputShape
  }
}

/**
  Resizes the input texture to a specific size. The input is expected to have
  3 or 4 channels. Always outputs a 3-channel image.
*/
public class Resize: Layer {
  let width: Int
  let height: Int
  var lanczos: MPSImageLanczosScale!

  public init(width: Int, height: Int, name: String = "") {
    self.width = width
    self.height = height
    super.init(name: name)
    allowsIncompleteShape = true
    wantsTextures = true
  }

  override public var typeName: String {
    return "Resize"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: width, height: height, channels: 3)
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {
    return lanczos = MPSImageLanczosScale(device: device)
  }

  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              sourceTexture: MTLTexture,
                              destinationTensor: Tensor) {
    lanczos.encode(commandBuffer: commandBuffer,
                   sourceTexture: sourceTexture,
                   destinationTexture: destinationTensor.image!.texture)
    destinationTensor.written(byLayer: self)
  }

  /**
    Crops the input image before it gets scaled down.

    The crop region is specified in input image coordinates.

    If you're always cropping the same region you can call this method right
    before or after compiling the model. If you're always cropping a different
    region (for example, using face detection on the input texture) then you
    should call this method right before you encode the model.
  */
  public func setCropRect(x: Double, y: Double, width: Double, height: Double) {
    let scaleX = Double(self.width) / width
    let scaleY = Double(self.height) / height
    let translateX = -x * scaleX
    let translateY = -y * scaleY
    var transform = MPSScaleTransform(scaleX: scaleX,
                                      scaleY: scaleY,
                                      translateX: translateX,
                                      translateY: translateY)

    withUnsafePointer(to: &transform) { ptr in
      lanczos.scaleTransform = ptr
    }
  }

  public func setCropRect(_ rect: CGRect) {
    setCropRect(x: Double(rect.origin.x),
                y: Double(rect.origin.y),
                width: Double(rect.width),
                height: Double(rect.height))
  }
}

/**
  The Custom layer type accepts any object that conforms to this protocol.

  - NOTE: The `encode()` function must do the following:
  
          // Let Metal know the temporary image can be recycled.
          if let image = sourceImage as? MPSTemporaryImage {
            image.readCount -= 1
          }
*/
public protocol CustomKernel {
  func encode(commandBuffer: MTLCommandBuffer, sourceImage: MPSImage, destinationImage: MPSImage)
}

/**
  Use this to call your own compute kernels.
*/
public class Custom: Layer {
  let custom: CustomKernel
  let width: Int?
  let height: Int?
  let channels: Int?

  /**
    Creates a new layer using a custom compute kernel.

    - Note: If `width`, `height`, or `channels` is nil, then that dimension
      from the input shape is passed through unchanged.
  */
  public init(_ custom: CustomKernel,
              width: Int? = nil,
              height: Int? = nil,
              channels: Int? = nil,
              name: String = "") {
    self.custom = custom
    self.width = width
    self.height = height
    self.channels = channels
    super.init(name: name)

    // If the output shape is completely specified, then this layer accepts
    // any input, even if some dimensions are unknown.
    if width != nil && height != nil && channels != nil {
      allowsIncompleteShape = true
    }
  }

  override public var typeName: String {
    return "Custom"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
    return DataShape(width: width ?? inputShape.width,
                     height: height ?? inputShape.height,
                     channels: channels ?? inputShape.channels)
  }

  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {
    custom.encode(commandBuffer: commandBuffer,
                  sourceImage: sourceTensor.image!,
                  destinationImage: destinationTensor.image!)
    destinationTensor.written(byLayer: self)
  }
}

public class DepthwiseConvolution: Layer {
  let kernel: (Int, Int)
  let channelMultiplier: Int
  let stride: (Int, Int)
  let activation: MPSCNNNeuron?
//  var compute: DepthwiseConvolutionKernel!
  var compute: MPSCNNConvolution!

  /**
    Creates a depth-wise convolution layer.
    
    Currently only supports .same padding.
  
    - Parameters:
      - kernel: `(width, height)`
      - stride: `(x, y)`
      - useReLU: Whether to apply a ReLU directly in the shader. You can also
        add `Activation(relu)` behind this layer instead.
      - name: The name is used to load the layer's parameters.
  */
  public init(kernel: (Int, Int),
              channelMultiplier: Int = 1,
              stride: (Int, Int) = (1, 1),
              activation: MPSCNNNeuron? = nil,
              useBias: Bool = true,
              name: String = "") {
    self.kernel = kernel
    self.stride = stride
    self.activation = activation
    self.channelMultiplier = channelMultiplier
    super.init(name: name, useBias: useBias)
  }

  override public var typeName: String {
    return "DepthwConv"
  }

  override public func outputShape(for inputShape: DataShape) -> DataShape {
      return DataShape(width: (inputShape.width - 1)  / stride.0 + 1,
                      height: (inputShape.height - 1) / stride.1 + 1,
                    channels: inputShape.channels * channelMultiplier)
  }

  override public func weightCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return inputShape.channels * kernel.1 * kernel.0 * channelMultiplier
  }

  public override func biasCount(inputShape: DataShape, outputShape: DataShape) -> Int {
    return useBias ? outputShape.channels : 0
  }

  override public func createCompute(device: MTLDevice,
                                     inputShape: DataShape,
                                     outputShape: DataShape,
                                     weights: ParameterData?,
                                     biases: ParameterData?) throws {

    guard let weights = weights else {
      throw ModelError.compileError(message: "missing weights for layer '\(name)'")
    }

    var biasTerms: UnsafeMutablePointer<Float>?
    if useBias {
      guard let biases = biases else {
        throw ModelError.compileError(message: "missing bias terms for layer '\(name)'")
      }
      biasTerms = biases.pointer
    }

    /*
    compute = DepthwiseConvolutionKernel(device: device,
                                         kernelWidth: kernel.0,
                                         kernelHeight: kernel.1,
                                         featureChannels: inputShape.channels,
                                         strideInPixelsX: stride.0,
                                         strideInPixelsY: stride.1,
                                         channelMultiplier: 1,
                                         neuronFilter: activation,
                                         kernelWeights: weights.pointer,
                                         biasTerms: biasTerms)
     */

    // TODO: MPS's depthwise convolution has the weights in a different
    // order, so transpose them. I will change the API for this class so
    // that it uses the same weights order as MPS, so that on iOS 10 it
    // will use Forge's kernel but on 11 it uses the MPS kernel (which is
    // faster).
    /*
    let convCount = inputShape.channels * kernel.0 * kernel.1
    var convWeights = [Float](repeating: 0, count: convCount)
    let mpsChanStride = kernel.0 * kernel.1
    let mpsHeightStride = kernel.1
    let mpsWidthStride = 1
    let forgeChanStride = 1
    let forgeHeightStride = inputShape.channels * kernel.0
    let forgeWidthStride = inputShape.channels
    for c in 0..<inputShape.channels {
      for h in 0..<kernel.1 {
        for w in 0..<kernel.0 {
          convWeights[c*mpsChanStride + h*mpsHeightStride + w*mpsWidthStride] = weights.pointer[c*forgeChanStride + h*forgeHeightStride + w*forgeWidthStride]
        }
      }
    }
    */
    if #available(iOS 11,*) {
    let desc = MPSCNNDepthWiseConvolutionDescriptor(kernelWidth: kernel.0,
                                                    kernelHeight: kernel.1,
                                                    inputFeatureChannels: inputShape.channels,
                                                    outputFeatureChannels: inputShape.channels * self.channelMultiplier,
                                                    neuronFilter: activation)
    desc.strideInPixelsX = stride.0
    desc.strideInPixelsY = stride.1

    compute = MPSCNNConvolution(device: device,
                                convolutionDescriptor: desc,
                                //kernelWeights: convWeights,
                                kernelWeights: weights.pointer,
                                biasTerms: biasTerms,
                                flags: .none)
    }
    compute.edgeMode = .zero
  }

  override public func encode(commandBuffer: MTLCommandBuffer,
                              sourceTensor: Tensor,
                              destinationTensor: Tensor) {

    
    compute.offset = offsetForConvolution(padding: .same,
                                          sourceWidth: sourceTensor.shape.width,
                                          sourceHeight: sourceTensor.shape.height,
                                          destinationWidth: destinationTensor.shape.width,
                                          destinationHeight: destinationTensor.shape.height,
                                          kernelWidth: kernel.0,
                                          kernelHeight: kernel.1,
                                          strideInPixelsX: stride.0,
                                          strideInPixelsY: stride.1)

    compute.encode(commandBuffer: commandBuffer,
                   sourceImage: sourceTensor.image!,
                   destinationImage: destinationTensor.image!)
    
    destinationTensor.written(byLayer: self)
  }
}

public class PointwiseConvolution: Convolution {
  /**
    Creates a point-wise convolution layer, which is really the same as a 
    convolutional layer with a 1x1 kernel.
  */
  public init(channels: Int,
              stride: (Int, Int) = (1, 1),
              activation: MPSCNNNeuron? = nil,
              useBias: Bool = true,
              name: String = "") {
    super.init(kernel: (1, 1), channels: channels, activation: activation,
               useBias: useBias, name: name)
  }

  override public var typeName: String {
    return "PointwConv"
  }
}
