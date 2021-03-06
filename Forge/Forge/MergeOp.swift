/*
 Copyright (c) 2016-2017 Pavel Mayer
 
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

import Metal
import MetalPerformanceShaders
import Accelerate


/**
 These values get passed to the compute kernel.
 */
public struct MergeParams {
  var inputImages: Int16 = 0
  var inputSlicesPerImage: Int16 = 0
  var destSliceOffset: Int16 = 0
}

public enum MergeOpType: Int16 {
  case Add = 1, Multiply, Maximum, Average
}


/**
 Merge all images inside an MPSImage applying an op like sum, max, mult or avrg
 */
public class MergeOpKernel {
  
  let device: MTLDevice
  let pipeline_array: MTLComputePipelineState
  var pipeline_single: MTLComputePipelineState? = nil
  var params = MergeParams()

  /**
   Creates a new MergeOpKernel object.
   
   - Parameters:
   - featureChannels: The number of channels in the input image. The output
   image will have the same number of channels.

   */
  
  var featureChannels : Int
  var inputFeatureImages : Int
  var inputSlicesPerImage : Int

  public init(device: MTLDevice,
              inputFeatureImages: Int,
              featureChannels: Int,
              featureOp: MergeOpType) {
    
    self.device = device
    self.inputFeatureImages = inputFeatureImages
    self.featureChannels = featureChannels
    self.inputSlicesPerImage = (featureChannels + 3)/4

    let constants = MTLFunctionConstantValues()
    var op = ushort(featureOp.rawValue)
    constants.setConstantValue(&op, type: .ushort, withName: "opType")

    // If there's only one texture slice in the image, we have might
    // use kernel that uses texture2d objects as output, depending on the output image
    if featureChannels <= 4 {
      pipeline_single = makeFunction(device: device, name: "mergeImages", constantValues: constants, useForgeLibrary: true)
    }
    pipeline_array = makeFunction(device: device, name: "mergeImages_array", constantValues: constants, useForgeLibrary: true)
  }
  
  public func encode(commandBuffer: MTLCommandBuffer,
                     sourceImage: MPSImage,
                     destinationImage: MPSImage,
                     destinationChannelOffset: Int,
                     destinationImageNumber: Int) {
    precondition(destinationChannelOffset % 4 == 0)
    precondition(destinationChannelOffset < self.featureChannels)
    precondition(destinationImageNumber < destinationImage.numberOfImages)
    if let encoder = commandBuffer.makeComputeCommandEncoder() {
      let pipeline = (featureChannels <= 4 && destinationImage.numberOfImages == 1) ? pipeline_single! : pipeline_array

      encoder.setComputePipelineState(pipeline)
      encoder.setTexture(sourceImage.texture, index: 0)
      encoder.setTexture(destinationImage.texture, index: 1)
      self.params.inputImages = Int16(sourceImage.numberOfImages);
      self.params.inputSlicesPerImage = Int16((sourceImage.featureChannels + 3)/4);
      let destSliceOffset = self.inputSlicesPerImage * destinationImageNumber + destinationChannelOffset/4
      self.params.destSliceOffset = Int16(destSliceOffset)
      encoder.setBytes(&params, length: MemoryLayout<MergeParams>.size, index: 0)
      
      encoder.dispatch(pipeline: pipeline,
                       width: destinationImage.width,
                       height: destinationImage.height,
                       featureChannels: self.featureChannels,
                       numberOfImages: 1)
      
      encoder.endEncoding()
    }
    
    if let image = sourceImage as? MPSTemporaryImage {
      image.readCount -= 1
    }
  }
}

