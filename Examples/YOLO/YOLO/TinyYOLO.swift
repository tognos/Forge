//
//  TinyYOLO.swift
//  YOLO
//
//  Created by Pavel Mayer on 25.08.17.
//  Copyright © 2017 Tognos GmbH. All rights reserved.
//

import Foundation

import MetalPerformanceShaders
import Forge


/*
 The tiny-yolo-voc network from YOLOv2. https://pjreddie.com/darknet/yolo/
 
 This implementation is cobbled together from the following sources:
 
 - https://github.com/pjreddie/darknet
 - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowYoloDetector.java
 - https://github.com/allanzelener/YAD2K
 */

class TinyYOLO: YOLO {
  typealias PredictionType = YOLO.Prediction

  let tiny_anchors: [Float] = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
  
  public init(device: MTLDevice, inflightBuffers: Int) {
    // Note: YOLO expects the input pixels to be in the range 0-1. Our input
    // texture most likely has pixels with values 0-255. However, since Forge
    // uses .float16 as the channel format the Resize layer will automatically
    // convert the pixels to be between 0 and 1.

    let leaky = MPSCNNNeuronReLU(device: device, a: 0.1)
      
      let inputWidth = 416
      let inputHeight = 416
      let input = Input()
      
      let output = input
        --> Resize(width: inputWidth, height: inputHeight)
        --> Convolution(kernel: (3, 3), channels: 16, activation: leaky, name: "conv1")
        --> MaxPooling(kernel: (2, 2), stride: (2, 2))
        --> Convolution(kernel: (3, 3), channels: 32, activation: leaky, name: "conv2")
        --> MaxPooling(kernel: (2, 2), stride: (2, 2))
        --> Convolution(kernel: (3, 3), channels: 64, activation: leaky, name: "conv3")
        --> MaxPooling(kernel: (2, 2), stride: (2, 2))
        --> Convolution(kernel: (3, 3), channels: 128, activation: leaky, name: "conv4")
        --> MaxPooling(kernel: (2, 2), stride: (2, 2))
        --> Convolution(kernel: (3, 3), channels: 256, activation: leaky, name: "conv5")
        --> MaxPooling(kernel: (2, 2), stride: (2, 2))
        --> Convolution(kernel: (3, 3), channels: 512, activation: leaky, name: "conv6")
        --> MaxPooling(kernel: (2, 2), stride: (1, 1), padding: .same)
        --> Convolution(kernel: (3, 3), channels: 1024, activation: leaky, name: "conv7")
        --> Convolution(kernel: (3, 3), channels: 1024, activation: leaky, name: "conv8")
        --> Convolution(kernel: (1, 1), channels: 125, activation: nil, name: "conv9")
      
      let model = Model(input: input, output: output)
      
      let success = model.compile(device: device, inflightBuffers: inflightBuffers) {
        name, count, type in ParameterLoaderBundle(name: name,
                                                   count: count,
                                                   suffix: type == .weights ? "_W" : "_b",
                                                   ext: "bin")
      }
      super.init(model: model,
                 inputWidth: inputWidth, inputHeight: inputHeight,
                 blockSize: 32,
                 gridHeight: 13, gridWidth: 13,
                 boxesPerCell: 5,
                 numClasses: 20,
                 anchors: tiny_anchors,
                 threshold: 0.5)
    
      if success {
        print(model.summary())
      }
  }
 }
