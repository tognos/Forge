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
import QuartzCore

/*
  A simple DSL for building neural networks without the boilerplate.
*/

precedencegroup ChainPrecedence {
  associativity: left
  higherThan: MultiplicationPrecedence
}

infix operator --> : ChainPrecedence

public func --> (lhs: Tensor, rhs: Layer) -> Tensor {
  return Tensor(input: lhs, layer: rhs)
}

public typealias ParameterCallback = (String, Int, ParameterType) -> ParameterData?

public enum ModelError: Error {
  case compileError(message: String)
}

public protocol TensorDumper {
  func dump(tensor: Tensor) -> Bool
}

/**
  The top-level object for the neural network.
  
  How to build and use the model:
  
  1. first create layers and tensors,
  2. instantiate a `Model` using the input and output tensor,
  3. call `compile()` to construct all the Metal objects,
  4. use `summary()` to verify that your model is correct,
  5. use `encode()` to perform inference.
*/

public class Model {
  let input: Tensor
  let output: Tensor
  public var debugTrace = false

  public class ImageInfo {
    init(owners : Set<Tensor> = [], writtenCount : Int = 0) {
      self.owners = owners
      self.writtenCount = writtenCount
    }
    public var owners : Set<Tensor>
    public var writtenCount : Int
    public var description: String {
      get {
        var line = "written " + writtenCount.description + ", owners:"
        for tensor in owners {
          line += tensor.shortId + " "
        }
        return line
      }
    }
  }
  
  var compiled = false
  var tensors: [Tensor] = []
  var imageDescriptors: [DataShape: MPSImageDescriptor] = [:]
  var imageDescriptorList: [MPSImageDescriptor] = []
  var outputImages: [Tensor: [MPSImage]] = [:]
  var imageInfos: [MPSImage: ImageInfo] = [:]
  var numLayers = 0
  var totalParams = 0

  // Used during compiling.
  var device: MTLDevice!
  var parameterCallback: ParameterCallback!
  var inflightBuffers = 0

  // Only used when the very first layer accepts a texture instead of an image.
  var firstLayerEatsTexture = false
  var firstComputeLayer = true
  var sourceTexture: MTLTexture?

  public init(input: Tensor, output: Tensor) {
    self.input = input
    self.output = output
  }
  
  var imageInfosDescription : String {
    get {
      var line = ""
      for (image, info) in imageInfos {
        line.append(image.description + ":" + info.description + "\n")
      }
      return line
    }
  }

  /**
    Creates all the MPSCNN objects for this graph.
    
    - Parameters:
      - inflightBuffers: How many tasks the CPU and GPU can do in parallel.
      - parameterCallback: Used for loading the parameters of the layers.
        The closure takes three arguments: name, expected parameter count, and
        whether to load the weights or bias values for the layer.
  */
  public func compile(device: MTLDevice,
                      inflightBuffers: Int = 3,
                      parameterCallback: @escaping ParameterCallback) -> Bool {
    if compiled {
      print("Compile error: graph has already been compiled")
      return false
    }

    let startTime = CACurrentMediaTime()

    self.device = device
    self.inflightBuffers = inflightBuffers
    self.parameterCallback = parameterCallback

    do {
      tensors = topologicalSort(from: input)

      try completeGraph()
      try createComputeForAllLayers()

      imageDescriptorList = Array(imageDescriptors.values)
      for imageDesc in imageDescriptorList {
        imageDesc.storageMode = .private
      }

      let elapsed = CACurrentMediaTime() - startTime
      print("Compiling took \(elapsed) seconds")

      compiled = true
      return true

    } catch ModelError.compileError(let message) {
      print("Compile error:", message)
      return false

    } catch {
      print("Unknown error: \(error)")
      return false
    }
  }

  /**
    This gives us an array of tensors in an order that is guaranteed to be
    correct (but possibly is not the order the model was specified in).
  */
  func topologicalSort(from source: Tensor) -> [Tensor] {
    var stack = [Tensor]()
    var visited = Set<Tensor>()

    func depthFirstSearch(_ source: Tensor) {
      for neighbor in source.next {
        if !visited.contains(neighbor) {
          depthFirstSearch(neighbor)
        }
      }
      stack.append(source)
      visited.insert(source)
    }

    depthFirstSearch(source)
    return stack.reversed()
  }

  /**
    Makes sure the graph can actually be compiled and fills in any missing
    information. This also fills up the cache with MPSImageDescriptors.
  */
  func completeGraph() throws {
    for (i, tensor) in tensors.enumerated() {
      tensor.model = self
      if let layer = tensor.layer {
        numLayers += 1

        // Assign a name to any layers that don't have one.
        if layer.name.isEmpty {
          layer.name = "__\(layer.typeName)_\(i+1)__"
        }

        // If the layer expects a fully-specified shape but the previous layers
        // didn't fill in the width/height/depth, then we cannot continue.
        if let input = tensor.input, !input.shape.isFullySpecified
                                  && !layer.allowsIncompleteShape {
          throw ModelError.compileError(message: "input shape \(input.shape) for layer '\(layer)' has unknown dimensions")
        }
      }

      if tensor.shape.isFullySpecified {
        registerImageDescriptor(for: tensor.shape)
      }
    }
  }

  func registerImageDescriptor(for shape: DataShape) {
    if imageDescriptors[shape] == nil {
      imageDescriptors[shape] = shape.createImageDescriptor()
    }
  }

  /**
    Creates compute kernels for the layers in the graph. Also allocates any
    non-temporary MPSImages for tensors that want them.
  */
  func createComputeForAllLayers() throws {
    for tensor in tensors {
      if let layer = tensor.layer, let input = tensor.input {
        // Only make the compute once for each layer (important for layers
        // that get reused).
        if !layer.createdCompute {
          try createCompute(for: layer, input: input, output: tensor)
          layer.createdCompute = true
        }
      }

      // Make an MPSImage for any tensor that asks for a real image instead
      // of a temporary one. We keep track of these in a dictionary.
      if !tensor.imageIsTemporary {
        addOutputImage(for: tensor)
      }
    }

    // Always make an MPSImage for the last tensor.
    if let output = tensors.last {
      addOutputImage(for: output)
    }
  }

  /**
    Creates Metal objects for the specified layer.
  */
  func createCompute(for layer: Layer, input: Tensor, output: Tensor) throws {
    // FUTURE: Sort the layers by largest weights to smallest, in order to
    // load the largest layers first. This makes it possible to load very
    // big models on devices with limited memory capacity, since the params
    // need to be copied into MPSCNN and therefore are in memory twice for
    // a short while.

    //print("createCompute:", input, "-->", output)

    var weightParameters: ParameterData?
    let weightCount = layer.weightCount(inputShape: input.shape, outputShape: output.shape)
    if weightCount > 0 {
      totalParams += weightCount
      weightParameters = parameterCallback(layer.name, weightCount, .weights)
    }

    var biasParameters: ParameterData?
    let biasCount = layer.biasCount(inputShape: input.shape, outputShape: output.shape)
    if biasCount > 0 {
      totalParams += biasCount
      biasParameters = parameterCallback(layer.name, biasCount, .biases)
    }
    // workaround for a bug where in a release build the runtime deinits the bias
    // parameters before the are used
    try withExtendedLifetime(biasParameters,
                         {try layer.createCompute(device: device,
                              inputShape: input.shape,
                              outputShape: output.shape,
                              weights: weightParameters,
                              biases: biasParameters)}
    )

    layer.paramCount = weightCount + biasCount
    //print("createCompute: done", input, "-->", output, "biasParameters = \(biasParameters?.pointer)")
    // Does the first layer take a MTLTexture or an MPSImage?
    if firstComputeLayer {
      if layer.wantsTextures { firstLayerEatsTexture = true }
      firstComputeLayer = false
    }
  }
  
  func written(image: MPSImage) {
    imageInfos[image]!.writtenCount += 1
    //print("Written to image \(image), info: \(imageInfos[image]!.description)")
  }
  func addOwner(for owner: Tensor, of image: MPSImage) {
    //print("Adding owner \(owner) for image \(image)")
    if imageInfos[image] != nil {
      if imageInfos[image]!.owners.contains(owner) {
        fatalError("tensor \(owner) already owns image \(image)")
      }
      imageInfos[image]!.owners.insert(owner)
    } else {
      imageInfos[image] = ImageInfo(owners: Set<Tensor>([owner]))
    }
  }

  func removeOwner(for tensor: Tensor, of image: MPSImage) {
    if imageInfos[image] != nil {
      if !imageInfos[image]!.owners.contains(tensor) {
        fatalError("tensor (\tensor) does not ows image \(self)")
      }
      imageInfos[image]!.owners.remove(tensor)
    } else {
      fatalError("tensor (\tensor) does not own any image)")
    }
  }
  func addOutputImage(for tensor: Tensor) {
    //print("adding output image for tensor", tensor)
    /*
    guard let imgDesc = imageDescriptors[tensor.shape] else {
      fatalError("Error: could not find image descriptor for shape \(tensor.shape)")
    }
     */
    // create a new image descriptor for output images that need to have a storage mode
    // .shared because we want to access them with the CPU
    let imgDesc = tensor.shape.createImageDescriptor()
    imgDesc.storageMode = .shared
    // Since the GPU can be working on several inputs at once, we need to
    // allocate multiple images.
    var array: [MPSImage] = []
    for _ in 0..<inflightBuffers {
      let image = MPSImage(device: device, imageDescriptor: imgDesc)
      array.append(image)
    }
    outputImages[tensor] = array
  }
  func removeOutputImage(for tensor: Tensor) {
    outputImages.removeValue(forKey: tensor)
  }

  /**
    Returns a summary of all the layers and tensors in the model, including
    their types, shapes, and number of parameters. Useful for making sure your
    model is correct.
  */
  public func summary() -> String {
    guard compiled else { return "(Model is not compiled.)" }

    var s = ""
    s += "Layer/Tensor                   Type       Output Shape     Parameters    In/Out\n"
    s += "--------------------------------------------------------------------------------\n"

    for tensor in tensors {
      s += tensor.summary() + "\n"
    }

    s += "--------------------------------------------------------------------------------\n"
    s += "Number of layers: \(numLayers) (tensors: \(tensors.count))\n"
    s += "Total parameters: \(totalParams)\n"
    //s += debugGraph()
    return s
  }
  public func debugSummary(current: Tensor, marker: String, debugTensor: Tensor?) -> String {
    guard compiled else { return "(Model is not compiled.)" }
    
    var s = ""
    s += "Layer/Tensor                   Type       Output Shape     Parameters    trc In/Out image                            w r x cur\n"
    s += "------------------------------------------------------------------------------------------------------------------------------\n"

    for tensor in tensors {
      s += tensor.debugSummary(isCurrent: tensor === current, marker: marker) + "\n"
      if debugTensor != nil && tensor == debugTensor {
        break
      }
    }
    
    s += "------------------------------------------------------------------------------------------------------------------------------\n"

    return s
  }
  
  // Prints a dot description of the model
  // Install dot and graphviz, copy the output to a file "fileName.dot"
  // and use the following command to create a .png image of the the graph:
  // dot -Tpng -o fileName.png fileName.dot
  public func debugGraph() -> String {
    guard compiled else { return "(Model is not compiled.)" }
    
    var s = "\n"
    s += "# Install dot and graphviz, copy to fileName.dot and convert with 'dot -Tpng -o fileName.png fileName.dot'\n"
    s += "digraph G {\n"
    s += "concentrate=True;\n"
    s += "rankdir=TB;\n"
    s += "node [shape=record];\n"
    
    for tensor in tensors {
      let layerName = tensor.layer?.name ?? tensor.shortId
      let layerType = tensor.layer?.typeName ?? "Tensor"
      s += "  \(layerName) [label=\"\(layerName) : \(layerType)\\n\(tensor.shape.debugDescription)\"];\n"
      for out in tensor.next {
        let outName = out.layer?.name ?? out.shortId
        s += "  " + layerName + " -> " + outName + ";\n"
      }
    }
    
    s += "}\n"

    return s
  }

  func reset() {
    //print("Clearing imageInfos")
    for tensor in tensors {
      tensor.release()
      tensor.readCount = tensor.next.count
      tensor.releasedReadCount = nil
    }
    imageInfos = [:]
  }
  func clearTempImages() {
    //print("Clearing temp images")
    for tensor in tensors {
      tensor.deleteTempImageForced()
    }
  }
  /**
   Encodes the GPU commands for a forward pass of the neural network.
   */

  public func encode(commandBuffer: MTLCommandBuffer, texture: MTLTexture?, inflightIndex: Int, mpsImage : MPSImage? = nil, debugOutput : Tensor? = nil) {
    //let startTime = CACurrentMediaTime()
    //print("====== Starting model encode ======, inflightIndex =",inflightIndex)
    
    
    if !compiled {
      print("Error: graph has not been compiled yet")
      return
    }
    reset()

    //print("Initial imageInfos:")
    //print(imageInfosDescription)
    
    //MPSTemporaryImage.prefetchStorage(with: commandBuffer,
    //                                  imageDescriptorList: imageDescriptorList)
    var addedDebugOut = false
    if debugOutput != nil {
      // We are running in debug mode, so clear old output images
      if outputImages[debugOutput!] == nil && debugOutput != tensors[0] {
        addOutputImage(for: debugOutput!)
        addedDebugOut = true
      }
    }
    
    var sourceImage: MPSImage?
    if (mpsImage == nil) {
      if firstLayerEatsTexture && (debugOutput != nil && debugOutput != tensors[0]) {
        sourceTexture = texture
      } else {
        sourceImage = MPSImage(texture: texture!, featureChannels: 3)
      }
    } else {
      precondition(texture == nil, "use either texture or mpsImage as input, but not both")
      sourceImage = mpsImage
    }
    
    tensors[0].image = sourceImage
    
    for tensor in tensors {
      if debugTrace {
        print("Model: encoding for", tensor.debugDescription)
        print(self.debugSummary(current: tensor, marker: "<--", debugTensor: debugOutput))
      }
      
      encode(tensor: tensor, commandBuffer: commandBuffer, inflightIndex: inflightIndex, debugTensor: debugOutput)
      
      if debugTrace {
        print("Model: done encoding for", tensor.debugDescription)
      }
      if debugOutput != nil && tensor == debugOutput {
        if debugTrace {
          print("Reached debug output tensor", tensor.shortId)
        }
        if addedDebugOut {
          // remove output image so it is a temporary image in the next run
          removeOutputImage(for: debugOutput!)
        }
        if debugTrace {
          print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        }
        return
      }
    }

    //let elapsed = CACurrentMediaTime() - startTime
    //print("Encoding took \(elapsed) seconds")
  }
  
  func encode(tensor: Tensor, commandBuffer: MTLCommandBuffer, inflightIndex: Int, debugTensor: Tensor?) {
    // If a tensor does not have a layer (true for Input and Concatenate), then 
    // pass through the source image unchanged.
    guard let layer = tensor.layer else {
      //print("tensor does not belong to a layer, tensor = "+tensor.debugDescription)
      return
    }
    
    // If the tensor has a real MPSImage, use that. Otherwise make a temp one.
    func createImage(for tensor: Tensor) -> MPSImage {
      if debugTrace { print("createImage for "+tensor.debugDescription) }
      if let images = outputImages[tensor] {
        let storedImage = images[inflightIndex]
        if debugTrace {
          print("createImage returning outputImage[tensor=\(tensor.shortId)][inflightIndex=\(inflightIndex)], image=\(storedImage)")
        }
        return storedImage
      } else {
        guard let desc = imageDescriptors[tensor.shape] else {
          fatalError("Error: no image descriptor found for shape \(tensor.shape)")
        }
        let image = MPSTemporaryImage(commandBuffer: commandBuffer,
                                      imageDescriptor: desc)
        // We have to use the image's readCount value to maintain both read and write counts
        // add also number of inputs to bump up readcount with the writecount for this tensor
        image.readCount = tensor.readCount + tensor.previous.count
        if tensor.typeName == "Collect" {
          // for collect tensors, all the outputs of our inputs will also read this image
          for input in tensor.previous {
            if debugTrace {
              print("Collect, raising image read count from \(image.readCount) by outputs of input \(input.shortId) = \(input.next.count)")
            }
            image.readCount += input.next.count
            // when an input is a concat tensor, all its inputs will directly write to our image,
            // and all their outputs would read from our image, but concat inputs can not have
            // multiple outputs, we ensure that during initialization
            if input.typeName == "Concat" && input.destinationTensor != nil {
              if debugTrace {
                print("Concat --> Collect, raising image read count from \(image.readCount) by # of concat inputs = \(input.previous.count)")
              }
              image.readCount += input.previous.count
            }
          }
        }
        if debugTrace {
          print("createImage returning new temp image \(image.description) with readcount \(image.readCount) (\(tensor.readCount)+\(tensor.previous.count)) for shape \(tensor.shape)")
        }
        return image
      }
    }
    
    // If this tensor does not use its own image, then grab the one from the
    // destination tensor. Otherwise, make a new image.
    if let destTensor = tensor.destinationTensor {
      if let image = destTensor.image {
        if debugTrace {
          print("Tensor",tensor.shortId,"has destinationTensor",destTensor.shortId,", using dest's existing image: \(image)")
        }
        tensor.image = image
      } else {

        destTensor.image = createImage(for: destTensor)
        if debugTrace {
          print("Tensor",tensor.shortId,"has destinationTensor",destTensor.shortId," but no image, creating new for dest and itself")
        }
        if let tmpImage = destTensor.image as? MPSTemporaryImage {
          tmpImage.readCount += 1
          if debugTrace {
          print("Tensor",tensor.shortId,"has destinationTensor",destTensor.shortId," but no image, got new tmp image for dest and itself w. read count",tmpImage.readCount)
          }
        }
        tensor.image = destTensor.image
      }
    } else {
      tensor.image = createImage(for: tensor)
      if debugTrace {
        if let tmpImage = tensor.image as? MPSTemporaryImage {
          print("Tensor",tensor.shortId,"has no destinationTensor, created new own w. read count",tmpImage.readCount, "outputs:",tensor.next.count)
        } else {
          print("Tensor",tensor.shortId,"has no destinationTensor, created non-temporary image")
        }
      }
    }
    
    guard let inputTensor = tensor.input else {
      fatalError("Error: missing source tensor")
    }
    
    if layer.wantsTextures {
      //print("layer wants texture")
      let inputTexture: MTLTexture
      if let sourceImage = inputTensor.image {
        inputTexture = sourceImage.texture
      } else if let sourceTexture = sourceTexture {
        inputTexture = sourceTexture   // valid only for first layer
      } else {
        fatalError("Error: layer '\(layer.name)' expected source texture")
      }
      
      if debugTrace {
        print(self.debugSummary(current: tensor, marker: "<*>", debugTensor: debugTensor))
      }
      layer.encode(commandBuffer: commandBuffer,
                   sourceTensor: inputTensor,
                   sourceTexture: inputTexture,
                   destinationTensor: tensor)
      
      if let image = inputTensor.image as? MPSTemporaryImage {
        //print("decrementing readcount for input image, current = \(image.readCount)")
        image.readCount -= 1
      }
    } else {
      if imageInfos[inputTensor.image!]!.writtenCount < inputTensor.previous.count {
        fatalError("Error: input tensor \(inputTensor.shortId) has not received all its inputs yet, image=\(String(describing: inputTensor.image?.description)) written=\(imageInfos[inputTensor.image!]!.writtenCount), needed=\(inputTensor.previous.count)")
      }
      //print("1)inputTensor.image=",inputTensor.image.debugDescription)
      if debugTrace {
        print(self.debugSummary(current: tensor, marker: "<->", debugTensor: debugTensor))
      }
      //print(self.imageInfosDescription)
      //print("2)inputTensor.image=",inputTensor.image.debugDescription)

      layer.encode(commandBuffer: commandBuffer,
                   sourceTensor: inputTensor,
                   destinationTensor: tensor)
    }


    // At this point we've used the image from the sourceTensor, and should
    // decrement its reference count. When it hits 0, we nil out its `image`
    // property so that a new MPSTemporaryImage will be allocated on the next
    // pass through the network.
    
    inputTensor.release(byLayer: layer)
    if debugTrace {
      print(self.debugSummary(current: tensor, marker: "-->", debugTensor: debugTensor))
    }

    //print("Model.encode Tensor done\n")
    //print(self.imageInfosDescription)

  }

  func images(for tensor: Tensor) -> [MPSImage] {
    return outputImages[tensor] ?? []
  }

  /** 
    Returns the output from the given tensor. This tensor must have its
    `imageIsTemporary` property set to false!
  */
  public func image(for tensor: Tensor, inflightIndex: Int) -> MPSImage {
    return images(for: tensor)[inflightIndex]
  }

  /** Returns the output from the last tensor in the model. */
  public func outputImage(inflightIndex: Int) -> MPSImage {
    return image(for: output, inflightIndex: inflightIndex)
  }
  // Runs the model synchronously and returns the result
  // This function is mainly intended for testing and debugging and
  // should not be used in production code where performance is desired
  public func evaluate(commandQueue: MTLCommandQueue, device : MTLDevice, input: MPSImage?, inputTexure: MTLTexture? = nil, debugOutput : Tensor? = nil) -> MPSImage {
    
    //print("inputTexure:",inputTexure.debugDescription)
    autoreleasepool {
      guard let commandBuffer = commandQueue.makeCommandBuffer() else { fatalError("can't make command buffer") }
      self.encode(commandBuffer: commandBuffer, texture: inputTexure, inflightIndex: 0, mpsImage: input, debugOutput: debugOutput)
      commandBuffer.commit()
      self.clearTempImages()
      commandBuffer.waitUntilCompleted()
    }
    if debugOutput == nil {
      return self.outputImage(inflightIndex: 0)
    } else {
      return debugOutput!.image!
    }
  }

  public func debugEvaluate(commandQueue: MTLCommandQueue, device : MTLDevice, input: MPSImage?, inputTexure: MTLTexture? = nil, dumper: TensorDumper) -> Void {
    for tensor in tensors {
      var ok = true
      for out in tensor.next {
        if out.previous.count > 1 {
          ok = false
        }
      }
      if ok {
        if debugTrace {
          print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
          print("Running up to tensor: "+tensor.shortId)
        }
        let _ = evaluate(commandQueue: commandQueue, device: device, input: input, inputTexure: inputTexure, debugOutput: tensor)
        precondition(dumper.dump(tensor: tensor), "dumper failed")
        if debugTrace {
          print("ImageInfos after evaluating:\n"+imageInfosDescription)
        }
        self.clearTempImages()
        self.reset() // make sure our output image will be properly cleared
      } else {
        print("Skipping tensor "+tensor.shortId+" because destination tensor has multiple inputs")
      }
    }
  }
}

