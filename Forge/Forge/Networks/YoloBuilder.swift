
//
//  Begin of autogenerated swift source code
//
//  YoloBuilder.swift
//
//  created 2017-10-11 20:39 by keras2metal.py
//
//  Converter wittenn by Pavel Mayer, Tognos GmbH, http://tognos.com/
//  based on YADK and Forge yolo2metal.py
//

import Foundation
//import Forge
import MetalPerformanceShaders

open class YoloBuilder : NetworkBuilder {

private(set) public var model: Model
private(set) public var device: MTLDevice
private(set) public var name = "yolo"

public required init(device: MTLDevice) {
  self.device = device

let leaky = MPSCNNNeuronReLU(device: device, a: 0.10000)
let input_scale = MPSCNNNeuronLinear(device: device, a: 2, b: -1)
let input = Input()
let input_1 = input --> Resize(width: 608, height: 608) --> Activation(input_scale, name: "input_scale")
let conv2d_1 = Convolution(kernel: (3, 3), channels: 32, activation: leaky, name: "conv2d_1")
let max_pooling2d_1 = MaxPooling(kernel: (2, 2), stride: (2, 2), padding: .same, name: "max_pooling2d_1")
let conv2d_2 = Convolution(kernel: (3, 3), channels: 64, activation: leaky, name: "conv2d_2")
let max_pooling2d_2 = MaxPooling(kernel: (2, 2), stride: (2, 2), padding: .same, name: "max_pooling2d_2")
let conv2d_3 = Convolution(kernel: (3, 3), channels: 128, activation: leaky, name: "conv2d_3")
let conv2d_4 = Convolution(kernel: (1, 1), channels: 64, activation: leaky, name: "conv2d_4")
let conv2d_5 = Convolution(kernel: (3, 3), channels: 128, activation: leaky, name: "conv2d_5")
let max_pooling2d_3 = MaxPooling(kernel: (2, 2), stride: (2, 2), padding: .same, name: "max_pooling2d_3")
let conv2d_6 = Convolution(kernel: (3, 3), channels: 256, activation: leaky, name: "conv2d_6")
let conv2d_7 = Convolution(kernel: (1, 1), channels: 128, activation: leaky, name: "conv2d_7")
let conv2d_8 = Convolution(kernel: (3, 3), channels: 256, activation: leaky, name: "conv2d_8")
let max_pooling2d_4 = MaxPooling(kernel: (2, 2), stride: (2, 2), padding: .same, name: "max_pooling2d_4")
let conv2d_9 = Convolution(kernel: (3, 3), channels: 512, activation: leaky, name: "conv2d_9")
let conv2d_10 = Convolution(kernel: (1, 1), channels: 256, activation: leaky, name: "conv2d_10")
let conv2d_11 = Convolution(kernel: (3, 3), channels: 512, activation: leaky, name: "conv2d_11")
let conv2d_12 = Convolution(kernel: (1, 1), channels: 256, activation: leaky, name: "conv2d_12")
let conv2d_13 = Convolution(kernel: (3, 3), channels: 512, activation: leaky, name: "conv2d_13")
let max_pooling2d_5 = MaxPooling(kernel: (2, 2), stride: (2, 2), padding: .same, name: "max_pooling2d_5")
let conv2d_14 = Convolution(kernel: (3, 3), channels: 1024, activation: leaky, name: "conv2d_14")
let conv2d_15 = Convolution(kernel: (1, 1), channels: 512, activation: leaky, name: "conv2d_15")
let conv2d_16 = Convolution(kernel: (3, 3), channels: 1024, activation: leaky, name: "conv2d_16")
let conv2d_17 = Convolution(kernel: (1, 1), channels: 512, activation: leaky, name: "conv2d_17")
let conv2d_18 = Convolution(kernel: (3, 3), channels: 1024, activation: leaky, name: "conv2d_18")
let conv2d_19 = Convolution(kernel: (3, 3), channels: 1024, activation: leaky, name: "conv2d_19")
let conv2d_21 = Convolution(kernel: (1, 1), channels: 64, activation: leaky, name: "conv2d_21")
let conv2d_20 = Convolution(kernel: (3, 3), channels: 1024, activation: leaky, name: "conv2d_20")
let space_to_depth_x2 = SpaceToDepthX2(name: "space_to_depth_x2")
let conv2d_22 = Convolution(kernel: (3, 3), channels: 1024, activation: leaky, name: "conv2d_22")
let conv2d_23 = Convolution(kernel: (1, 1), channels: 425, name: "conv2d_23")

do {
let conv2d_13 = input_1 --> conv2d_1 --> max_pooling2d_1 --> conv2d_2 --> max_pooling2d_2
         --> conv2d_3 --> conv2d_4 --> conv2d_5 --> max_pooling2d_3 --> conv2d_6
         --> conv2d_7 --> conv2d_8 --> max_pooling2d_4 --> conv2d_9 --> conv2d_10
         --> conv2d_11 --> conv2d_12 --> conv2d_13
let space_to_depth_x2 = conv2d_13 --> conv2d_21 --> space_to_depth_x2
let conv2d_20 = conv2d_13 --> max_pooling2d_5 --> conv2d_14 --> conv2d_15 --> conv2d_16
         --> conv2d_17 --> conv2d_18 --> conv2d_19 --> conv2d_20
let concatenate_1 = Concatenate([space_to_depth_x2, conv2d_20], name: "concatenate_1")
let conv2d_23 = concatenate_1 --> conv2d_22 --> conv2d_23
let output = conv2d_23
model = Model(input: input, output: output)
}
} // init
public func compile(inflightBuffers: Int) -> Bool {
return model.compile(device: device, inflightBuffers: inflightBuffers) { 
  name, count, type in ParameterLoaderBundle(name: name,
  count: count,
  prefix: "yolo-",
  suffix: type == .weights ? ".weights" : ".biases",
  ext: "bin")
}
} // func
} // class

// end of autogenerated forge net generation code
