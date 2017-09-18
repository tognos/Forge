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

public struct PadParams {
    var paddingTop : UInt16 = 0
    var paddingBottom : UInt16 = 0
    var paddingLeft : UInt16 = 0
    var paddingRight : UInt16 = 0
    var paddingValue : Float16 = 0
    var destSliceOffset : UInt16 = 0
};
/**
 Pads all channels with a value
 */
public class PadKernel {
    
    let device: MTLDevice
    let pipeline: MTLComputePipelineState
    var params = PadParams()
    
    /**
     Creates a new MergeOpKernel object.
     
     - Parameters:
     - featureChannels: The number of channels in the input image. The output
     image will have the same number of channels.
     - permute: A list of channels to permute. (The same channel index is
     allowed to appear more than once.)
     */
    var featureChannels : Int

    public init(device: MTLDevice,
                tblr_padding: (Int, Int, Int, Int),
                padValue: Float,
                featureChannels: Int,
                writesToArray : Bool) {
        
        self.device = device
        self.featureChannels = featureChannels

        self.params.paddingTop = UInt16(tblr_padding.0)
        self.params.paddingBottom = UInt16(tblr_padding.1)
        self.params.paddingLeft = UInt16(tblr_padding.2)
        self.params.paddingRight = UInt16(tblr_padding.3)
        self.params.paddingValue = float16From32(padValue)
        
        // If there's more than one texture slice in the output image we have to use a
        // kernel that uses texture2d_array objects as output.
        let functionName: String
        if featureChannels <= 4 {
            if writesToArray {
                functionName = "pad_to_array"
            } else {
                functionName = "pad"
            }
        } else {
            functionName = "pad_arrays"
        }
        pipeline = makeFunction(device: device, name: functionName, constantValues: nil, useForgeLibrary: true)
    }
    
    public func encode(commandBuffer: MTLCommandBuffer,
                       sourceImage: MPSImage,
                       destinationImage: MPSImage,
                       destinationSliceOffset: Int) {
        if let encoder = commandBuffer.makeComputeCommandEncoder() {
            encoder.setComputePipelineState(pipeline)
            encoder.setTexture(sourceImage.texture, index: 0)
            encoder.setTexture(destinationImage.texture, index: 1)
            
            self.params.destSliceOffset = UInt16(destinationSliceOffset)
            encoder.setBytes(&params, length: MemoryLayout<PadParams>.size, index: 0)
            
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


