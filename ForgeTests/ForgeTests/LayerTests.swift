//
//  TestUtils.swift
//  ForgeTests
//
//  Created by Pavel Mayer on 28.09.17.
//  Copyright © 2017 Tognos GmbH. All rights reserved.
//

import Foundation
import Forge
import MetalPerformanceShaders

class LayerTests {
  
  func testActivationLayer() {
    print("\(self).\(#function)")
    
    //f(x) = a * x + b
    let linear = MPSCNNNeuronLinear(device: device, a: 3.0, b: 1.0)
    let input = Input(width: 3, height: 2, channels: 1)
    let activation = Activation(linear, name: "activation")
    
    let output = input --> activation
    
    let model = Model(input: input, output: output)
    
    let success = model.compile(device: device, inflightBuffers: 1) {
      name, count, type in ParameterLoaderFromMemory(name: name,
                                                     count: count,
                                                     weights: [:],
                                                     suffix: type == .weights ? ".weights" : ".biases")
    }
    if success {
      print(model.summary())
    }
    
    let testInputData : [[[[Float]]]] = [[[[1,2,3],
                                           [4,5,6]]]]
    let testInputImage = MPSImage(device: device, images: testInputData)
    let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
    
    //print("resultImage=",resultImage)
    let resultData = resultImage.toFloatArray4D()
    //print("resultData=",resultData)
    
    let expectedOutputData : [[[[Float]]]] = [[[[4,7,10],
                                                [13,16,19]]]]
    if !areAlmostEqual(resultData, expectedOutputData, maxError: Float(1e-2), reportUnequal: printLocation) {
      fatalError("Assertion failed: \(resultData) not equal to \(expectedOutputData)")
    }
  }
  
  func testZeroPaddingLayer_1ch() {
    print("\(self).\(#function)")
    
    let input = Input(width: 1, height: 1, channels: 1)
    
    let output = input --> ZeroPadding(tblr_padding: (1,2,3,4), name: "zero_padding")
    
    let model = Model(input: input, output: output)
    
    let success = model.compile(device: device, inflightBuffers: 1) {
      name, count, type in ParameterLoaderFromMemory(name: name,
                                                     count: count,
                                                     weights: [:],
                                                     suffix: type == .weights ? ".weights" : ".biases")
    }
    if success {
      print(model.summary())
    }
    
    let testInputData : [[[[Float]]]] = [[[[1]]]]
    let testInputImage = MPSImage(device: device, images: testInputData)
    let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
    
    print("resultImage=",resultImage)
    let resultData = resultImage.toFloatArray4D()
    print("resultData=",resultData)
    
    let expectedOutputData : [[[[Float]]]] = [[[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]]
    if !areAlmostEqual(resultData, expectedOutputData, maxError: Float(1e-2), reportUnequal: printLocation) {
      fatalError("Assertion failed: \(resultData) not equal to \(expectedOutputData)")
    }
  }
  func testZeroPaddingLayer_3ch() {
    print("\(self).\(#function)")
    
    let input = Input(width: 2, height: 2, channels: 3)
    
    let output = input --> ZeroPadding(tblr_padding: (1,2,3,4), name: "zero_padding")
    
    let model = Model(input: input, output: output)
    
    let success = model.compile(device: device, inflightBuffers: 1) {
      name, count, type in ParameterLoaderFromMemory(name: name,
                                                     count: count,
                                                     weights: [:],
                                                     suffix: type == .weights ? ".weights" : ".biases")
    }
    if success {
      print(model.summary())
    }
    
    let testInputData : [[[[Float]]]] = [[[[11,12],
                                           [13,14]],
                                          [[21,22],
                                           [23,24]],
                                          [[31,32],
                                           [33,34]]]]
    let testInputImage = MPSImage(device: device, images: testInputData)
    print("testInputImage:", testInputImage.debugDescription)
    let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
    
    print("resultImage=",resultImage)
    let resultData = resultImage.toFloatArray4D()
    print("resultData=",resultData)
    
    let expectedOutputData : [[[[Float]]]] = [[[[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 11, 12, 0, 0, 0, 0],
                                                [0, 0, 0, 13, 14, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                               [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 21, 22, 0, 0, 0, 0],
                                                [0, 0, 0, 23, 24, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                               [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 31, 32, 0, 0, 0, 0],
                                                [0, 0, 0, 33, 34, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
    if !areAlmostEqual(resultData, expectedOutputData, maxError: Float(1e-2), reportUnequal: printLocation) {
      fatalError("Assertion failed: \(resultData) not equal to \(expectedOutputData)")
    }
  }
  
  func testZeroPaddingLayer_3ch_Image() {
    print("\(self).\(#function)")
    /*
     let input = Input(width: 1, height: 1, channels: 3)
     
     let output = input --> ZeroPadding(tblr_padding: (1,2,3,4), name: "zero_padding")
     */
    let input = Input()
    let output = input --> Resize(width: 224, height: 224) --> ZeroPadding(tblr_padding: (0,0,0,0), name: "zero_padding")
    let model = Model(input: input, output: output)
    
    let success = model.compile(device: device, inflightBuffers: 1) {
      name, count, type in ParameterLoaderFromMemory(name: name,
                                                     count: count,
                                                     weights: [:],
                                                     suffix: type == .weights ? ".weights" : ".biases")
    }
    if success {
      print(model.summary())
    }
    
    let testInputImage = loadTexture(named: "final0-224.jpg")
    //let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
    /*
     let dumper = DocumentDirectoryDumper(filePrefix: "resnet_50")
     model.debugEvaluate(commandQueue: commandQueue, device: device, input: nil, inputTexure: testInputImage!, dumper: dumper)
     if !dumper.shapes.saveAsJson(fileName: "shapes.json") {
     print("Error dumping shapes")
     }
     */
    /*
     let testInputData : [[[[Float]]]] = [[[[1]],[[2]],[[3]]]]
     let testInputImage = MPSImage(device: device, images: testInputData)
     */
    print("testInputImage:", testInputImage.debugDescription)
    let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: nil, inputTexure: testInputImage!)
    
    print("resultImage=",resultImage)
    let resultData = resultImage.toFloatArray4D()
    print("resultData=",resultData)
    
    let expectedOutputData : [[[[Float]]]] = [[[[0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 1, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0]],
                                               [[0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 2, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0]],
                                               [[0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 3, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0, 0]]]]
    if !areAlmostEqual(resultData, expectedOutputData, maxError: Float(1e-2), reportUnequal: printLocation) {
      fatalError("Assertion failed: \(resultData) not equal to \(expectedOutputData)")
    }
  }
  
  
  
  func testSpaceToDepthX2Layer() {
    print("\(self).\(#function)")
    
    let input = Input(width: 4, height: 4, channels: 4)
    
    let output = input --> SpaceToDepthX2(name: "space2DepthX2")
    
    let model = Model(input: input, output: output)
    
    let success = model.compile(device: device, inflightBuffers: 1) {
      name, count, type in ParameterLoaderFromMemory(name: name,
                                                     count: count,
                                                     weights: [:],
                                                     suffix: type == .weights ? ".weights" : ".biases")
    }
    if success {
      print(model.summary())
    }
    
    let testInputData : [[[[Float]]]] = [[[[1, 2, 3, 4 ],
                                           [5, 6, 7, 8 ],
                                           [9, 10,11,12],
                                           [13,14,15,16]],
                                          [[21, 22, 23, 24 ],
                                           [25, 26, 27, 28 ],
                                           [29, 210,211,212],
                                           [213,214,215,216]],
                                          [[31, 32, 33, 34 ],
                                           [35, 36, 37, 38 ],
                                           [39, 310,311,312],
                                           [313,314,315,316]],
                                          [[41, 42, 43, 44 ],
                                           [45, 46, 47, 48 ],
                                           [49, 410,411,412],
                                           [413,414,415,416]]]]
    let testInputImage = MPSImage(device: device, images: testInputData)
    let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
    
    print("resultImage=",resultImage)
    let resultData = resultImage.toFloatArray4D()
    print("resultData=",resultData)
    
    let expectedOutputData : [[[[Float]]]] = [[[[1, 2],
                                                [5, 6]],
                                               [[21, 22],
                                                [25, 26]],
                                               [[31, 32],
                                                [35, 36]],
                                               [[41, 42],
                                                [45, 46]],
                                               [[3, 4],
                                                [7, 8]],
                                               [[23, 24],
                                                [27, 28]],
                                               [[33, 34],
                                                [37, 38]],
                                               [[43, 44],
                                                [47, 48]],
                                               [[9, 10 ],
                                                [13, 14]],
                                               [[29, 210],
                                                [213, 214]],
                                               [[39, 310],
                                                [313, 314]],
                                               [[49, 410],
                                                [413, 414]],
                                               [[11, 12],
                                                [15, 16]],
                                               [[211, 212],
                                                [215, 216]],
                                               [[311, 312],
                                                [315, 316]],
                                               [[411, 412],
                                                [415, 416]]]]
    if !areAlmostEqual(resultData, expectedOutputData, maxError: Float(1e-2), reportUnequal: printLocation) {
      fatalError("Assertion failed: \(resultData) not equal to \(expectedOutputData)")
    }
  }
  func testCollect() {
    print("\(self).\(#function)")
    
    //f(x) = a * x + b
    let linear1 = MPSCNNNeuronLinear(device: device, a: 1.0, b: 0.0)
    let linear2 = MPSCNNNeuronLinear(device: device, a: 1.0, b: 0.0)
    let input = Input(width: 3, height: 2, channels: 4)
    let activation1 = Activation(linear1, name: "activation1")
    let activation2 = Activation(linear2, name: "activation2")
    
    do {
      let activation1 = input --> activation1
      let activation2 = input --> activation2
      let collected = Collect([activation1, activation2])
      let output = collected
      let model = Model(input: input, output: output)
      
      let success = model.compile(device: device, inflightBuffers: 1) {
        name, count, type in ParameterLoaderFromMemory(name: name,
                                                       count: count,
                                                       weights: [:],
                                                       suffix: type == .weights ? ".weights" : ".biases")
      }
      if success {
        print(model.summary())
      }
      
      // 1 2x3 Image with 4 channels
      let testInputData : [[[[Float]]]] = [[[[11,12,13],
                                             [14,15,16]],
                                            [[21,22,23],
                                             [24,25,26]],
                                            [[31,32,33],
                                             [34,35,36]],
                                            [[41,42,43],
                                             [44,45,46]]]]
      
      let testInputImage = MPSImage(device: device, images: testInputData)
      let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
      
      //print("resultImage=",resultImage)
      let resultData = resultImage.toFloatArray4D()
      //print("resultData=",resultData)
      
      let expectedOutputData : [[[[Float]]]] = [[[[11,12,13],
                                                  [14,15,16]],
                                                 [[21,22,23],
                                                  [24,25,26]],
                                                 [[31,32,33],
                                                  [34,35,36]],
                                                 [[41,42,43],
                                                  [44,45,46]]],
                                                [[[11,12,13],
                                                  [14,15,16]],
                                                 [[21,22,23],
                                                  [24,25,26]],
                                                 [[31,32,33],
                                                  [34,35,36]],
                                                 [[41,42,43],
                                                  [44,45,46]]]]
      if !areAlmostEqual(resultData, expectedOutputData, maxError: Float(1e-2), reportUnequal: printLocation) {
        fatalError("Assertion failed: \(resultData) not equal to \(expectedOutputData)")
      }
    }
  }
  
  func testSimpleMerge() {
    print("\(self).\(#function)")
    
    //f(x) = a * x + b
    let linear1 = MPSCNNNeuronLinear(device: device, a: 1.0, b: 0.0)
    let linear2 = MPSCNNNeuronLinear(device: device, a: 1.0, b: 1.0)
    let input = Input(width: 3, height: 2, channels: 4)
    let activation1 = Activation(linear1, name: "activation1")
    let activation2 = Activation(linear2, name: "activation2")
    
    do {
      let activation1 = input --> activation1
      let activation2 = input --> activation2
      let collected = Collect([activation1, activation2])
      let output = collected --> Add(name:"adder")
      let model = Model(input: input, output: output)
      
      let success = model.compile(device: device, inflightBuffers: 1) {
        name, count, type in ParameterLoaderFromMemory(name: name,
                                                       count: count,
                                                       weights: [:],
                                                       suffix: type == .weights ? ".weights" : ".biases")
      }
      if success {
        print(model.summary())
      }
      
      // 1 2x3 Image with 4 channels
      let testInputData : [[[[Float]]]] = [[[[11,12,13],
                                             [14,15,16]],
                                            [[21,22,23],
                                             [24,25,26]],
                                            [[31,32,33],
                                             [34,35,36]],
                                            [[41,42,43],
                                             [44,45,46]]]]
      
      let testInputImage = MPSImage(device: device, images: testInputData)
      let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
      
      //print("resultImage=",resultImage)
      let resultData = resultImage.toFloatArray4D()
      //print("resultData=",resultData)
      
      let expectedOutputData : [[[[Float]]]] = [[[[23, 25, 27],
                                                  [29, 31, 33]],
                                                 [[43, 45, 47],
                                                  [49, 51, 53]],
                                                 [[63, 65, 67],
                                                  [69, 71, 73]],
                                                 [[83, 85, 87],
                                                  [89, 91, 93]]]]
      if !areAlmostEqual(resultData, expectedOutputData, maxError: Float(1e-2), reportUnequal: printLocation) {
        fatalError("Assertion failed: \(resultData) not equal to \(expectedOutputData)")
      }
    }
  }
  func testComplexMerge() {
    print("\(self).\(#function)")
    
    //f(x) = a * x + b
    let linear1 = MPSCNNNeuronLinear(device: device, a: 1.0, b: 0.0)
    let linear2 = MPSCNNNeuronLinear(device: device, a: 1.0, b: 1.0)
    let input = Input(width: 3, height: 2, channels: 4)
    let activation1 = Activation(linear1, name: "activation1")
    let activation2 = Activation(linear2, name: "activation2")
    let activation3 = Activation(linear2, name: "activation3")
    let activation4 = Activation(linear2, name: "activation4")
    
    do {
      let activation1 = input --> activation1
      let activation3 = activation1 --> activation2 --> activation3
      let collected = Collect([activation1, activation3]) --> Add(name:"adder")
      let output = collected --> activation4
      let model = Model(input: input, output: output)
      
      let success = model.compile(device: device, inflightBuffers: 1) {
        name, count, type in ParameterLoaderFromMemory(name: name,
                                                       count: count,
                                                       weights: [:],
                                                       suffix: type == .weights ? ".weights" : ".biases")
      }
      if success {
        print(model.summary())
      }
      
      let testInputData : [[[[Float]]]] = [[[[11,12,13],
                                             [14,15,16]],
                                            [[21,22,23],
                                             [24,25,26]],
                                            [[31,32,33],
                                             [34,35,36]],
                                            [[41,42,43],
                                             [44,45,46]]]]
      
      let testInputImage = MPSImage(device: device, images: testInputData)
      let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
      
      //print("resultImage=",resultImage)
      let resultData = resultImage.toFloatArray4D()
      //print("resultData=",resultData)
      
      let expectedOutputData : [[[[Float]]]] = [[[[25, 27, 29],
                                                  [31, 33, 35]],
                                                 [[45, 47, 49],
                                                  [51, 53, 55]],
                                                 [[65, 67, 69],
                                                  [71, 73, 75]],
                                                 [[85, 87, 89],
                                                  [91, 93, 95]]]]
      if !areAlmostEqual(resultData, expectedOutputData, maxError: Float(1e-2), reportUnequal: printLocation) {
        fatalError("Assertion failed: \(resultData) not equal to \(expectedOutputData)")
      }
    }
  }
  func testMPSImages() {
    print("\(self).\(#function)")
    let maxError : Float = 1e-2
    // 1 channel
    do {
      let test_input_data_3x2_1c_1i : [[[[Float]]]] = [[[[1,2,3],
                                                         [4,5,6]]]]
      //      print("test_input_data_3x2_1c_1i:",test_input_data_3x2_1c_1i)
      let test_input_image_3x2_1c_1i = MPSImage(device: device, images: test_input_data_3x2_1c_1i)
      
      let test_read_data_3x2_1c_1i = test_input_image_3x2_1c_1i.toFloatArray4D()
      //      print("test_read_data_3x2_1c_1i:",test_read_data_3x2_1c_1i)
      
      if !(areAlmostEqual(test_input_data_3x2_1c_1i, test_read_data_3x2_1c_1i, maxError: maxError, reportUnequal: printLocation)) {
        fatalError("Assertion failed: \(test_input_data_3x2_1c_1i) not equal to \(test_read_data_3x2_1c_1i)")
      }
      
      print("shape3211=\(shape(test_input_data_3x2_1c_1i))")
      let linear = test_input_image_3x2_1c_1i.toFloatArrayChannelsInterleaved()
      let reshaped = linear.reshaped(test_input_image_3x2_1c_1i.shapeChannelsInterleaved)
      print("reshaped: \(reshaped)")
      let transposedData = transposed(reshaped, axes: (2,0,1,3))
      print("transposed: \(transposedData)")
    }
    // 2 channels
    do {
      let test_input_data_3x2_2c_1i : [[[[Float]]]] = [[[[101,102,103],
                                                         [104,105,106]],
                                                        [[201,202,203],
                                                         [204,205,206]]]]
      //      print("test_input_data_3x2_2c_1i:",test_input_data_3x2_2c_1i)
      let test_input_image_3x2_2c_1i = MPSImage(device: device, images: test_input_data_3x2_2c_1i)
      
      let test_read_data_3x2_2c_1i = test_input_image_3x2_2c_1i.toFloatArray4D()
      //      print("test_read_data_3x2_2c_1i:",test_read_data_3x2_2c_1i)
      
      if !(areAlmostEqual(test_input_data_3x2_2c_1i, test_read_data_3x2_2c_1i, maxError: maxError, reportUnequal: printLocation)) {
        fatalError("Assertion failed: \(test_input_data_3x2_2c_1i) not equal to \(test_read_data_3x2_2c_1i)")
      }
    }
    // 3 channels
    do {
      let test_input_data_3x2_3c_1i : [[[[Float]]]] = [[[[101,102,103],
                                                         [104,105,106]],
                                                        [[201,202,203],
                                                         [204,205,206]],
                                                        [[301,302,303],
                                                         [304,305,306]]]]
      print("test_input_data_3x2_3c_1i shape:\(shape(test_input_data_3x2_3c_1i))")
      //      print("test_input_data_3x2_3c_1i:",test_input_data_3x2_3c_1i)
      let test_input_image_3x2_3c_1i = MPSImage(device: device, images: test_input_data_3x2_3c_1i)
      
      let test_read_data_3x2_3c_1i = test_input_image_3x2_3c_1i.toFloatArray4D()
      //      print("test_read_data_3x2_3c_1i:",test_read_data_3x2_3c_1i)
      
      if !(areAlmostEqual(test_input_data_3x2_3c_1i, test_read_data_3x2_3c_1i, maxError: maxError, reportUnequal: printLocation)) {
        fatalError("Assertion failed: \(test_input_data_3x2_3c_1i) not equal to \(test_read_data_3x2_3c_1i)")
      }
      let linear = test_input_image_3x2_3c_1i.toFloatArrayChannelsInterleaved()
      let reshaped = linear.reshaped(test_input_image_3x2_3c_1i.shapeChannelsInterleaved)
      print("reshaped: \(reshaped)")
      let interleaved : [[[[Float]]]] = [[[[101, 201, 301], [102, 202, 302], [103, 203, 303]],
                                          [[104, 204, 304], [105, 205, 305], [106, 206, 306]]]]
      if !(areAlmostEqual(reshaped, interleaved, maxError: maxError, reportUnequal: printLocation)) {
        fatalError("Assertion failed: \(reshaped) not equal to \(interleaved)")
      }
      print("interleaved shape:\(shape(interleaved))")
      let transposedData = transposed(reshaped, axes: (0,3,1,2)) // transpose from channel interleaved to channels together
      print("transposed: \(transposedData)")

      if !(areAlmostEqual(test_input_data_3x2_3c_1i, transposedData, maxError: maxError, reportUnequal: printLocation)) {
        fatalError("Assertion failed: \(test_input_data_3x2_3c_1i) not equal to \(transposedData)")
      }
    }
    // 4 channels
    do {
      let test_input_data_3x2_4c_1i : [[[[Float]]]] = [[[[101,102,103],
                                                         [104,105,106]],
                                                        [[201,202,203],
                                                         [204,205,206]],
                                                        [[301,302,303],
                                                         [304,305,306]],
                                                        [[401,402,403],
                                                         [404,405,406]]]]
      //      print("test_input_data_3x2_4c_1i:",test_input_data_3x2_4c_1i)
      let test_input_image_3x2_4c_1i = MPSImage(device: device, images: test_input_data_3x2_4c_1i)
      
      let test_read_data_3x2_4c_1i = test_input_image_3x2_4c_1i.toFloatArray4D()
      //      print("test_read_data_3x2_4c_1i:",test_read_data_3x2_4c_1i)
      
      if !(areAlmostEqual(test_input_data_3x2_4c_1i, test_read_data_3x2_4c_1i, maxError: maxError, reportUnequal: printLocation)) {
        fatalError("Assertion failed: \(test_input_data_3x2_4c_1i) not equal to \(test_read_data_3x2_4c_1i)")
      }
    }
    // 5 channels
    do {
      let test_input_data_3x2_5c_1i : [[[[Float]]]] = [[[[101,102,103],
                                                         [104,105,106]],
                                                        [[201,202,203],
                                                         [204,205,206]],
                                                        [[301,302,303],
                                                         [304,305,306]],
                                                        [[401,402,403],
                                                         [404,405,406]],
                                                        [[501,502,503],
                                                         [504,505,506]]]]
      //      print("test_input_data_3x2_5c_1i:",test_input_data_3x2_5c_1i)
      let test_input_image_3x2_5c_1i = MPSImage(device: device, images: test_input_data_3x2_5c_1i)
      
      let test_read_data_3x2_5c_1i = test_input_image_3x2_5c_1i.toFloatArray4D()
      //      print("test_read_data_3x2_5c_1i:",test_read_data_3x2_5c_1i)
      
      if !(areAlmostEqual(test_input_data_3x2_5c_1i, test_read_data_3x2_5c_1i, maxError: maxError, reportUnequal: printLocation)) {
        fatalError("Assertion failed: \(test_input_data_3x2_5c_1i) not equal to \(test_read_data_3x2_5c_1i)")
      }
    }
    // 1 channel, 2 images
    do {
      let test_input_data_3x2_1c_2i : [[[[Float]]]] = [[[[11,12,13],
                                                         [14,15,16]]],
                                                       [[[21,22,23],
                                                         [24,25,26]]]]
      //      print("test_input_data_3x2_1c_2i:",test_input_data_3x2_1c_2i)
      let test_input_image_3x2_1c_2i = MPSImage(device: device, images: test_input_data_3x2_1c_2i)
      
      let test_read_data_3x2_1c_2i = test_input_image_3x2_1c_2i.toFloatArray4D()
      //      print("test_read_data_3x2_1c_2i:",test_read_data_3x2_1c_2i)
      
      if !(areAlmostEqual(test_input_data_3x2_1c_2i, test_read_data_3x2_1c_2i, maxError: maxError, reportUnequal: printLocation)) {
        fatalError("Assertion failed: \(test_input_data_3x2_1c_2i) not equal to \(test_read_data_3x2_1c_2i)")
      }
    }
    // 3 channels, 2 images
    do {
      let test_input_data_3x2_3c_2i : [[[[Float]]]] = [[[[111,112,113],
                                                         [114,115,116]],
                                                        [[121,122,123],
                                                         [124,125,126]],
                                                        [[131,132,133],
                                                         [134,135,136]]],
                                                       [[[211,212,213],
                                                         [214,215,216]],
                                                        [[221,222,223],
                                                         [224,225,226]],
                                                        [[231,232,233],
                                                         [234,235,236]]]]
      //      print("test_input_data_3x2_3c_2i:",test_input_data_3x2_3c_2i)
      let test_input_image_3x2_3c_2i = MPSImage(device: device, images: test_input_data_3x2_3c_2i)
      
      let test_read_data_3x2_3c_2i = test_input_image_3x2_3c_2i.toFloatArray4D()
      //      print("test_read_data_3x2_3c_2i:",test_read_data_3x2_3c_2i)
      
      if !(areAlmostEqual(test_input_data_3x2_3c_2i, test_read_data_3x2_3c_2i, maxError: maxError, reportUnequal: printLocation)) {
        fatalError("Assertion failed: \(test_input_data_3x2_3c_2i) not equal to \(test_read_data_3x2_3c_2i)")
      }
    }
    // 4 channels, 2 images
    do {
      let test_input_data_3x2_4c_2i : [[[[Float]]]] = [[[[111,112,113],
                                                         [114,115,116]],
                                                        [[211,212,213],
                                                         [214,215,216]],
                                                        [[311,312,313],
                                                         [314,315,316]],
                                                        [[411,412,413],
                                                         [414,415,416]]],
                                                       [[[121,122,123],
                                                         [124,125,126]],
                                                        [[221,222,223],
                                                         [224,225,226]],
                                                        [[321,322,323],
                                                         [324,325,326]],
                                                        [[421,422,423],
                                                         [424,425,426]]]]
      //      print("test_input_data_3x2_4c_2i:",test_input_data_3x2_4c_2i)
      let test_input_image_3x2_4c_2i = MPSImage(device: device, images: test_input_data_3x2_4c_2i)
      
      let test_read_data_3x2_4c_2i = test_input_image_3x2_4c_2i.toFloatArray4D()
      //      print("test_read_data_3x2_4c_2i:",test_read_data_3x2_4c_2i)
      
      if !(areAlmostEqual(test_input_data_3x2_4c_2i, test_read_data_3x2_4c_2i, maxError: maxError, reportUnequal: printLocation)) {
        fatalError("Assertion failed: \(test_input_data_3x2_4c_2i) not equal to \(test_read_data_3x2_4c_2i)")
      }
    }
    // 5 channels, 2 images
    do {
      let test_input_data_3x2_5c_2i : [[[[Float]]]] = [[[[111,112,113],
                                                         [114,115,116]],
                                                        [[211,212,213],
                                                         [214,215,216]],
                                                        [[311,312,313],
                                                         [314,315,316]],
                                                        [[411,412,413],
                                                         [414,415,416]],
                                                        [[511,512,513],
                                                         [514,515,516]]],
                                                       [[[121,122,123],
                                                         [124,125,126]],
                                                        [[221,222,223],
                                                         [224,225,226]],
                                                        [[321,322,323],
                                                         [324,325,326]],
                                                        [[421,422,423],
                                                         [424,425,426]],
                                                        [[521,522,523],
                                                         [524,525,526]]]]
      //      print("test_input_data_3x2_5c_2i:",test_input_data_3x2_5c_2i)
      let test_input_image_3x2_5c_2i = MPSImage(device: device, images: test_input_data_3x2_5c_2i)
      
      let test_read_data_3x2_5c_2i = test_input_image_3x2_5c_2i.toFloatArray4D()
      //      print("test_read_data_3x2_5c_2i:",test_read_data_3x2_5c_2i)
      
      if !(areAlmostEqual(test_input_data_3x2_5c_2i, test_read_data_3x2_5c_2i, maxError: maxError, reportUnequal: printLocation)) {
        fatalError("Assertion failed: \(test_input_data_3x2_5c_2i) not equal to \(test_read_data_3x2_5c_2i)")
      }
    }
  }
  
  
  func testSubtractMean() {
    print("\(self).\(#function)")
    
    let input = Input(width: 1, height: 1, channels: 3)
    let subtract_mean = Custom(SubtractMeanColor(device:device, red: 123.68, green: 116.779, blue: 103.939, scale: 255.0), name: "subtract_mean")
    let output = input --> subtract_mean
    
    let model = Model(input: input, output: output)
    
    let success = model.compile(device: device, inflightBuffers: 1) {
      name, count, type in ParameterLoaderFromMemory(name: name,
                                                     count: count,
                                                     weights: [:],
                                                     suffix: type == .weights ? ".weights" : ".biases")
    }
    if success {
      print(model.summary())
    }
    
    let testInputData : [[[[Float]]]] = [[[[123.68/255.0]],[[116.779/255.0]],[[103.939/255.0]]]]
    let testInputImage = MPSImage(device: device, images: testInputData)
    print("testInputImage:", testInputImage.debugDescription)
    let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
    
    print("resultImage=",resultImage)
    let resultData = resultImage.toFloatArray4D()
    print("resultData=",resultData)
    
    let expectedOutputData : [[[[Float]]]] = [[[[0]],[[0]],[[0]]]]
    if !areAlmostEqual(resultData, expectedOutputData, maxError: Float(0.1), reportUnequal: printLocation) {
      fatalError("Assertion failed: \(resultData) not equal to \(expectedOutputData)")
    }
  }
  
  
  func testResNet(debug: Bool) {
    print("\(self).\(#function)")
    
    var model : Model
    // begin of autogenerated forge net generation code
    
    let relu = MPSCNNNeuronReLU(device: device, a: 0)
    let input = Input()
    let swap_channels = Custom(TransposeChannelsKernel(device: device, featureChannels: 3, permute: [2,1,0]), name: "rgb2bgr")
    let subtract_mean = Custom(SubtractMeanColor(device:device, red: 123.68, green: 116.779, blue: 103.939, scale: 255.0), name: "subtract_mean")
    let input_2 = input --> Resize(width: 224, height: 224) -->  swap_channels -->  subtract_mean
    let zero_padding2d_1 = ZeroPadding(tblr_padding: (3, 3, 3, 3), name: "zero_padding2d_1")
    let conv1 = Convolution(kernel: (7, 7), channels: 64, stride: (2, 2), padding: .valid, activation: relu, name: "conv1")
    let max_pooling2d_1 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_1")
    let res2a_branch2a = Convolution(kernel: (1, 1), channels: 64, padding: .valid, activation: relu, name: "res2a_branch2a")
    let res2a_branch2b = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "res2a_branch2b")
    let res2a_branch2c = Convolution(kernel: (1, 1), channels: 256, padding: .valid, name: "res2a_branch2c")
    let res2a_branch1 = Convolution(kernel: (1, 1), channels: 256, padding: .valid, name: "res2a_branch1")
    let activation_4 = Activation(relu, name: "activation_4")
    let res2b_branch2a = Convolution(kernel: (1, 1), channels: 64, padding: .valid, activation: relu, name: "res2b_branch2a")
    let res2b_branch2b = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "res2b_branch2b")
    let res2b_branch2c = Convolution(kernel: (1, 1), channels: 256, padding: .valid, name: "res2b_branch2c")
    let activation_7 = Activation(relu, name: "activation_7")
    let res2c_branch2a = Convolution(kernel: (1, 1), channels: 64, padding: .valid, activation: relu, name: "res2c_branch2a")
    let res2c_branch2b = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "res2c_branch2b")
    let res2c_branch2c = Convolution(kernel: (1, 1), channels: 256, padding: .valid, name: "res2c_branch2c")
    let activation_10 = Activation(relu, name: "activation_10")
    let res3a_branch2a = Convolution(kernel: (1, 1), channels: 128, stride: (2, 2), padding: .valid, activation: relu, name: "res3a_branch2a")
    let res3a_branch2b = Convolution(kernel: (3, 3), channels: 128, activation: relu, name: "res3a_branch2b")
    let res3a_branch2c = Convolution(kernel: (1, 1), channels: 512, padding: .valid, name: "res3a_branch2c")
    let res3a_branch1 = Convolution(kernel: (1, 1), channels: 512, stride: (2, 2), padding: .valid, name: "res3a_branch1")
    let activation_13 = Activation(relu, name: "activation_13")
    let res3b_branch2a = Convolution(kernel: (1, 1), channels: 128, padding: .valid, activation: relu, name: "res3b_branch2a")
    let res3b_branch2b = Convolution(kernel: (3, 3), channels: 128, activation: relu, name: "res3b_branch2b")
    let res3b_branch2c = Convolution(kernel: (1, 1), channels: 512, padding: .valid, name: "res3b_branch2c")
    let activation_16 = Activation(relu, name: "activation_16")
    let res3c_branch2a = Convolution(kernel: (1, 1), channels: 128, padding: .valid, activation: relu, name: "res3c_branch2a")
    let res3c_branch2b = Convolution(kernel: (3, 3), channels: 128, activation: relu, name: "res3c_branch2b")
    let res3c_branch2c = Convolution(kernel: (1, 1), channels: 512, padding: .valid, name: "res3c_branch2c")
    let activation_19 = Activation(relu, name: "activation_19")
    let res3d_branch2a = Convolution(kernel: (1, 1), channels: 128, padding: .valid, activation: relu, name: "res3d_branch2a")
    let res3d_branch2b = Convolution(kernel: (3, 3), channels: 128, activation: relu, name: "res3d_branch2b")
    let res3d_branch2c = Convolution(kernel: (1, 1), channels: 512, padding: .valid, name: "res3d_branch2c")
    let activation_22 = Activation(relu, name: "activation_22")
    let res4a_branch2a = Convolution(kernel: (1, 1), channels: 256, stride: (2, 2), padding: .valid, activation: relu, name: "res4a_branch2a")
    let res4a_branch2b = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "res4a_branch2b")
    let res4a_branch2c = Convolution(kernel: (1, 1), channels: 1024, padding: .valid, name: "res4a_branch2c")
    let res4a_branch1 = Convolution(kernel: (1, 1), channels: 1024, stride: (2, 2), padding: .valid, name: "res4a_branch1")
    let activation_25 = Activation(relu, name: "activation_25")
    let res4b_branch2a = Convolution(kernel: (1, 1), channels: 256, padding: .valid, activation: relu, name: "res4b_branch2a")
    let res4b_branch2b = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "res4b_branch2b")
    let res4b_branch2c = Convolution(kernel: (1, 1), channels: 1024, padding: .valid, name: "res4b_branch2c")
    let activation_28 = Activation(relu, name: "activation_28")
    let res4c_branch2a = Convolution(kernel: (1, 1), channels: 256, padding: .valid, activation: relu, name: "res4c_branch2a")
    let res4c_branch2b = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "res4c_branch2b")
    let res4c_branch2c = Convolution(kernel: (1, 1), channels: 1024, padding: .valid, name: "res4c_branch2c")
    let activation_31 = Activation(relu, name: "activation_31")
    let res4d_branch2a = Convolution(kernel: (1, 1), channels: 256, padding: .valid, activation: relu, name: "res4d_branch2a")
    let res4d_branch2b = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "res4d_branch2b")
    let res4d_branch2c = Convolution(kernel: (1, 1), channels: 1024, padding: .valid, name: "res4d_branch2c")
    let activation_34 = Activation(relu, name: "activation_34")
    let res4e_branch2a = Convolution(kernel: (1, 1), channels: 256, padding: .valid, activation: relu, name: "res4e_branch2a")
    let res4e_branch2b = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "res4e_branch2b")
    let res4e_branch2c = Convolution(kernel: (1, 1), channels: 1024, padding: .valid, name: "res4e_branch2c")
    let activation_37 = Activation(relu, name: "activation_37")
    let res4f_branch2a = Convolution(kernel: (1, 1), channels: 256, padding: .valid, activation: relu, name: "res4f_branch2a")
    let res4f_branch2b = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "res4f_branch2b")
    let res4f_branch2c = Convolution(kernel: (1, 1), channels: 1024, padding: .valid, name: "res4f_branch2c")
    let activation_40 = Activation(relu, name: "activation_40")
    let res5a_branch2a = Convolution(kernel: (1, 1), channels: 512, stride: (2, 2), padding: .valid, activation: relu, name: "res5a_branch2a")
    let res5a_branch2b = Convolution(kernel: (3, 3), channels: 512, activation: relu, name: "res5a_branch2b")
    let res5a_branch2c = Convolution(kernel: (1, 1), channels: 2048, padding: .valid, name: "res5a_branch2c")
    let res5a_branch1 = Convolution(kernel: (1, 1), channels: 2048, stride: (2, 2), padding: .valid, name: "res5a_branch1")
    let activation_43 = Activation(relu, name: "activation_43")
    let res5b_branch2a = Convolution(kernel: (1, 1), channels: 512, padding: .valid, activation: relu, name: "res5b_branch2a")
    let res5b_branch2b = Convolution(kernel: (3, 3), channels: 512, activation: relu, name: "res5b_branch2b")
    let res5b_branch2c = Convolution(kernel: (1, 1), channels: 2048, padding: .valid, name: "res5b_branch2c")
    let activation_46 = Activation(relu, name: "activation_46")
    let res5c_branch2a = Convolution(kernel: (1, 1), channels: 512, padding: .valid, activation: relu, name: "res5c_branch2a")
    let res5c_branch2b = Convolution(kernel: (3, 3), channels: 512, activation: relu, name: "res5c_branch2b")
    let res5c_branch2c = Convolution(kernel: (1, 1), channels: 2048, padding: .valid, name: "res5c_branch2c")
    let activation_49 = Activation(relu, name: "activation_49")
    let avg_pool = AveragePooling(kernel: (7, 7), stride: (7, 7), name: "avg_pool")
    let fc1000 = Dense(neurons: 1000, activation: nil, name: "fc1000")
    
    do {
      let max_pooling2d_1 = input_2 --> zero_padding2d_1 --> conv1 --> max_pooling2d_1
      let res2a_branch1 = max_pooling2d_1 --> res2a_branch1
      let res2a_branch2c = max_pooling2d_1 --> res2a_branch2a --> res2a_branch2b --> res2a_branch2c
      
      let add_1 = Collect([res2a_branch2c, res2a_branch1], name: "for_add_1") --> Add(name: "add_1")
      let activation_4 = add_1 --> activation_4
      let res2b_branch2c = activation_4 --> res2b_branch2a --> res2b_branch2b --> res2b_branch2c
      let add_2 = Collect([res2b_branch2c, activation_4], name: "for_add_2") --> Add(name: "add_2")
      let activation_7 = add_2 --> activation_7
      let res2c_branch2c = activation_7 --> res2c_branch2a --> res2c_branch2b --> res2c_branch2c
      let add_3 = Collect([res2c_branch2c, activation_7], name: "for_add_3") --> Add(name: "add_3")
      let activation_10 = add_3 --> activation_10
      let res3a_branch1 = activation_10 --> res3a_branch1
      let res3a_branch2c = activation_10 --> res3a_branch2a --> res3a_branch2b --> res3a_branch2c
      let add_4 = Collect([res3a_branch2c, res3a_branch1], name: "for_add_4") --> Add(name: "add_4")
      let activation_13 = add_4 --> activation_13
      let res3b_branch2c = activation_13 --> res3b_branch2a --> res3b_branch2b --> res3b_branch2c
      let add_5 = Collect([res3b_branch2c, activation_13], name: "for_add_5") --> Add(name: "add_5")
      let activation_16 = add_5 --> activation_16
      let res3c_branch2c = activation_16 --> res3c_branch2a --> res3c_branch2b --> res3c_branch2c
      let add_6 = Collect([res3c_branch2c, activation_16], name: "for_add_6") --> Add(name: "add_6")
      let activation_19 = add_6 --> activation_19
      let res3d_branch2c = activation_19 --> res3d_branch2a --> res3d_branch2b --> res3d_branch2c
      let add_7 = Collect([res3d_branch2c, activation_19], name: "for_add_7") --> Add(name: "add_7")
      let activation_22 = add_7 --> activation_22
      let res4a_branch2c = activation_22 --> res4a_branch2a --> res4a_branch2b --> res4a_branch2c
      let res4a_branch1 = activation_22 --> res4a_branch1
      let add_8 = Collect([res4a_branch2c, res4a_branch1], name: "for_add_8") --> Add(name: "add_8")
      let activation_25 = add_8 --> activation_25
      let res4b_branch2c = activation_25 --> res4b_branch2a --> res4b_branch2b --> res4b_branch2c
      let add_9 = Collect([res4b_branch2c, activation_25], name: "for_add_9") --> Add(name: "add_9")
      let activation_28 = add_9 --> activation_28
      let res4c_branch2c = activation_28 --> res4c_branch2a --> res4c_branch2b --> res4c_branch2c
      let add_10 = Collect([res4c_branch2c, activation_28], name: "for_add_10") --> Add(name: "add_10")
      let activation_31 = add_10 --> activation_31
      let res4d_branch2c = activation_31 --> res4d_branch2a --> res4d_branch2b --> res4d_branch2c
      let add_11 = Collect([res4d_branch2c, activation_31], name: "for_add_11") --> Add(name: "add_11")
      let activation_34 = add_11 --> activation_34
      let res4e_branch2c = activation_34 --> res4e_branch2a --> res4e_branch2b --> res4e_branch2c
      let add_12 = Collect([res4e_branch2c, activation_34], name: "for_add_12") --> Add(name: "add_12")
      let activation_37 = add_12 --> activation_37
      let res4f_branch2c = activation_37 --> res4f_branch2a --> res4f_branch2b --> res4f_branch2c
      let add_13 = Collect([res4f_branch2c, activation_37], name: "for_add_13") --> Add(name: "add_13")
      let activation_40 = add_13 --> activation_40
      let res5a_branch2c = activation_40 --> res5a_branch2a --> res5a_branch2b --> res5a_branch2c
      let res5a_branch1 = activation_40 --> res5a_branch1
      let add_14 = Collect([res5a_branch2c, res5a_branch1], name: "for_add_14") --> Add(name: "add_14")
      let activation_43 = add_14 --> activation_43
      let res5b_branch2c = activation_43 --> res5b_branch2a --> res5b_branch2b --> res5b_branch2c
      let add_15 = Collect([res5b_branch2c, activation_43], name: "for_add_15") --> Add(name: "add_15")
      let activation_46 = add_15 --> activation_46
      let res5c_branch2c = activation_46 --> res5c_branch2a --> res5c_branch2b --> res5c_branch2c
      let add_16 = Collect([res5c_branch2c, activation_46], name: "for_add_16") --> Add(name: "add_16")
      let fc1000 = add_16 --> activation_49 --> avg_pool --> fc1000
      let output = fc1000 --> Softmax()
      model = Model(input: input, output: output)
    }
    var success = false
    let inflightBuffers = 1
    if debug {
      
      /*
       var weights : [String : [Float]] = [:]
       
       let identity_filter_plane : [[Float]] = [[0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,1,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0]]
       let zero_filter_plane : [[Float]] =     [[0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0]]
       let horizontal_filter_plane : [[Float]] = [[0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,1,2,1,0,0],
       [0,0,0,0,0,0,0],
       [0,0,-1,-2,-1,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0]]
       let vertical_filter_plane : [[Float]] = [[0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,1,0,-1,0,0],
       [0,0,2,0,-2,0,0],
       [0,0,1,0,-1,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0]]
       let edge_filter_plane : [[Float]] = [[0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,1,2,-1,0,0],
       [0,0,2,0,-2,0,0],
       [0,0,1,-2,-2,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0]]
       let blur_filter_plane : [[Float]] =     [[0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0],
       [0,0,1,0,-1,0,0],
       [0,0,2,0,-2,0,0],
       [0,0,1,0,-1,0,0],
       [0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0]]
       let zero_filter : [[[Float]]] = [zero_filter_plane, zero_filter_plane, zero_filter_plane]
       let identiy_filter : [[[Float]]] = [identity_filter_plane, identity_filter_plane, identity_filter_plane]
       let red_filter : [[[Float]]] = [identity_filter_plane, zero_filter_plane, zero_filter_plane]
       let green_filter : [[[Float]]] = [zero_filter_plane, identity_filter_plane, zero_filter_plane]
       let blue_filter : [[[Float]]] = [zero_filter_plane, zero_filter_plane, identity_filter_plane]
       let hor_edge_filter : [[[Float]]] = [horizontal_filter_plane, horizontal_filter_plane, horizontal_filter_plane]
       let vert_edge_filter : [[[Float]]] = [vertical_filter_plane, vertical_filter_plane, vertical_filter_plane]
       let edge_filter : [[[Float]]] = [edge_filter_plane, edge_filter_plane, edge_filter_plane]
       
       let filter_64 = [identiy_filter, zero_filter, red_filter, zero_filter, green_filter, zero_filter, blue_filter, zero_filter,
       zero_filter, hor_edge_filter, zero_filter, vert_edge_filter, zero_filter, edge_filter, zero_filter, identiy_filter,
       zero_filter, zero_filter, zero_filter, zero_filter, zero_filter, zero_filter, zero_filter, zero_filter,
       identiy_filter, identiy_filter,identiy_filter,identiy_filter,identiy_filter,identiy_filter,identiy_filter,identiy_filter,
       red_filter, red_filter, red_filter, red_filter, red_filter, red_filter, red_filter, red_filter,
       green_filter, green_filter, green_filter, green_filter, green_filter, green_filter, green_filter, green_filter,
       blue_filter, blue_filter, blue_filter, blue_filter, blue_filter, blue_filter, blue_filter, blue_filter,
       edge_filter, zero_filter,  edge_filter, zero_filter, edge_filter, zero_filter, edge_filter, zero_filter]
       /*
       var filter_64 = makeArray(dim: (64,3,7,7), value: Float(0.0))
       filter_64[0] = identiy_filter
       filter_64[1] = hor_edge_filter
       */
       
       /*
       //let test_weights = makeArray(dim: (64,7,7,3), value: 0)
       //let loadedWeightsShaped = loadedWeights.reshaped((3,7,7,64))
       //let transposedWeights = transposed(loadedWeightsShaped, axes: (3,2,1,0))
       //let flattenedWeights = flattened(transposedWeights)
       */
       /*
       print("should be (64,3,7,7): filter_64 shape=\(shape(filter_64))")
       let transposedWeights = transposed(filter_64, axes: (0,2,3,1))
       print("transposedWeights 64,7,7,3 shape=\(shape(transposedWeights))")
       let flattenedWeights = flattened(transposedWeights)
       */
       /*
       let testWeights = ParameterLoaderBundle(name: "res2b_branch2a", count: 16384, prefix: "resnet_50-",
       suffix: ".weights", ext: "bin")
       weights["res2b_branch2a.weights"] = Array<Float>(UnsafeBufferPointer(start: testWeights!.pointer, count: 16384))
       */
       
       let testWeights0 = ParameterLoaderBundle(name: "conv1", count: 9408, prefix: "resnet_50-",
       suffix: ".weights", ext: "bin")
       let loadedWeights = Array<Float>(UnsafeBufferPointer(start: testWeights0!.pointer, count: 9408))
       /*
       let loadedWeightsShaped = loadedWeights.reshaped((3,7,7,64))
       let transposedWeights = transposed(loadedWeightsShaped, axes: (3,1,2,0)) // or (3,2,1,0) --> 0,81 avrg diff 22,92 max diff
       let loadedWeightsShaped = loadedWeights.reshaped((3,7,7,64)) // best  --> 0,81 avrg
       */
       
       let loadedWeightsShaped = loadedWeights.reshaped((3,7,7,64))
       let transposedWeights = transposed(loadedWeightsShaped, axes: (3,1,2,0)) // or (3,2,1,0) --> 0,81 avrg diff 22,92 max diff
       //let transposedWeights = transposed(loadedWeightsShaped, axes: (3,2,1,0)) // or (3,1,2,0)
       let flattenedWeights = flattened(transposedWeights)
       
       // MPSCNN expects [outputChannels][kernelHeight][kernelWidth][inputChannels]
       //_ = flattenedWeights.saveInDocumentDirectory(fileName: "test_64_7_7_3_oc_h_w_ic.floats")
       _ = flattenedWeights.saveInDocumentDirectory(fileName: "test_conv1.floats")
       weights["conv1.weights"] = flattenedWeights
       //weights["conv1.biases"] = [Float](repeatElement(0, count: 64))
       
       
       success = model.compile(device: device, inflightBuffers: inflightBuffers) {
       name, count, type in ParameterLoaderBundleOrMemory(name: name,
       count: count,
       prefix: "resnet_50-",
       suffix: type == .weights ? ".weights" : ".biases",
       ext: "bin",
       weights: weights) }
       */
      
      success = model.compile(device: device, inflightBuffers: inflightBuffers) {
        name, count, type in ParameterLoaderBundle(name: name,
                                                   count: count,
                                                   prefix: "resnet_50-",
                                                   suffix: type == .weights ? ".weights" : ".biases",
                                                   ext: "bin") }
      
    } else {
      success = model.compile(device: device, inflightBuffers: 1) {
        name, count, type in ParameterLoaderRandom(count: count)
      }
    }
    // end of autogenerated forge net generation code
    
    if success {
      print(model.summary())
    }
    
    if debug {
      let testInputImage = loadTexture(named: "final1-224.jpg")
      //let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
      let dumper = DocumentDirectoryDumper(filePrefix: "resnet_50")
      model.debugEvaluate(commandQueue: commandQueue, device: device, input: nil, inputTexure: testInputImage!, dumper: dumper)
      if !dumper.shapes.saveAsJson(fileName: "shapes.json") {
        print("Error dumping shapes")
      }
      
      let probabilitiesImage = model.outputImage(inflightIndex: 0)
      let probabilities = probabilitiesImage.toFloatArray()
      assert(probabilities.count == 1000)
      print("probabilities: \(probabilitiesImage.toFloatArrayChannelsInterleaved())")
      
      typealias Prediction = (label: String, probability: Float, index: Int)
      var result = NeuralNetworkResult<Prediction>()
      result.predictions = probabilities.top(k: 5).map { x -> Prediction in (imageNetLabels[x.0], x.1, x.0) }
      print("predictions:\(result.predictions)")
    } else {
      let test_input_data_3x2_3c_1i : [[[[Float]]]] = [[[[11,12,13],
                                                         [14,15,16]],
                                                        [[21,22,23],
                                                         [24,25,26]],
                                                        [[31,32,33],
                                                         [34,35,36]]]]
      
      let testInputImage = MPSImage(device: device, images: test_input_data_3x2_3c_1i)
      let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
    }
  }
  
  func testInception_V3(debug: Bool) {
    print("\(self).\(#function)")
    
    var model : Model
    let relu = MPSCNNNeuronReLU(device: device, a: 0)
    let input_scale = MPSCNNNeuronLinear(device: device, a: 2, b: -1)
    let input = Input()
    let input_2 = input --> Resize(width: 299, height: 299) --> Activation(input_scale)
    let conv2d_1 = Convolution(kernel: (3, 3), channels: 32, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_1")
    let conv2d_2 = Convolution(kernel: (3, 3), channels: 32, padding: .valid, activation: relu, name: "conv2d_2")
    let conv2d_3 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_3")
    let max_pooling2d_1 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_1")
    let conv2d_4 = Convolution(kernel: (1, 1), channels: 80, padding: .valid, activation: relu, name: "conv2d_4")
    let conv2d_5 = Convolution(kernel: (3, 3), channels: 192, padding: .valid, activation: relu, name: "conv2d_5")
    let max_pooling2d_2 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_2")
    let conv2d_9 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_9")
    let conv2d_7 = Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "conv2d_7")
    let conv2d_10 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_10")
    let average_pooling2d_1 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_1")
    let conv2d_6 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_6")
    let conv2d_8 = Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "conv2d_8")
    let conv2d_11 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_11")
    let conv2d_12 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_12")
    let conv2d_16 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_16")
    let conv2d_14 = Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "conv2d_14")
    let conv2d_17 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_17")
    let average_pooling2d_2 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_2")
    let conv2d_13 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_13")
    let conv2d_15 = Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "conv2d_15")
    let conv2d_18 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_18")
    let conv2d_19 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_19")
    let conv2d_23 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_23")
    let conv2d_21 = Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "conv2d_21")
    let conv2d_24 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_24")
    let average_pooling2d_3 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_3")
    let conv2d_20 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_20")
    let conv2d_22 = Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "conv2d_22")
    let conv2d_25 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_25")
    let conv2d_26 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_26")
    let conv2d_28 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_28")
    let conv2d_29 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_29")
    let conv2d_27 = Convolution(kernel: (3, 3), channels: 384, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_27")
    let conv2d_30 = Convolution(kernel: (3, 3), channels: 96, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_30")
    let max_pooling2d_3 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_3")
    let conv2d_35 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_35")
    let conv2d_36 = Convolution(kernel: (1, 7), channels: 128, activation: relu, name: "conv2d_36")
    let conv2d_32 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_32")
    let conv2d_37 = Convolution(kernel: (7, 1), channels: 128, activation: relu, name: "conv2d_37")
    let conv2d_33 = Convolution(kernel: (7, 1), channels: 128, activation: relu, name: "conv2d_33")
    let conv2d_38 = Convolution(kernel: (1, 7), channels: 128, activation: relu, name: "conv2d_38")
    let average_pooling2d_4 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_4")
    let conv2d_31 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_31")
    let conv2d_34 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_34")
    let conv2d_39 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_39")
    let conv2d_40 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_40")
    let conv2d_45 = Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "conv2d_45")
    let conv2d_46 = Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "conv2d_46")
    let conv2d_42 = Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "conv2d_42")
    let conv2d_47 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_47")
    let conv2d_43 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_43")
    let conv2d_48 = Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "conv2d_48")
    let average_pooling2d_5 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_5")
    let conv2d_41 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_41")
    let conv2d_44 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_44")
    let conv2d_49 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_49")
    let conv2d_50 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_50")
    let conv2d_55 = Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "conv2d_55")
    let conv2d_56 = Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "conv2d_56")
    let conv2d_52 = Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "conv2d_52")
    let conv2d_57 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_57")
    let conv2d_53 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_53")
    let conv2d_58 = Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "conv2d_58")
    let average_pooling2d_6 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_6")
    let conv2d_51 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_51")
    let conv2d_54 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_54")
    let conv2d_59 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_59")
    let conv2d_60 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_60")
    let conv2d_65 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_65")
    let conv2d_66 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_66")
    let conv2d_62 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_62")
    let conv2d_67 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_67")
    let conv2d_63 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_63")
    let conv2d_68 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_68")
    let average_pooling2d_7 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_7")
    let conv2d_61 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_61")
    let conv2d_64 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_64")
    let conv2d_69 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_69")
    let conv2d_70 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_70")
    let conv2d_73 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_73")
    let conv2d_74 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_74")
    let conv2d_71 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_71")
    let conv2d_75 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_75")
    let conv2d_72 = Convolution(kernel: (3, 3), channels: 320, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_72")
    let conv2d_76 = Convolution(kernel: (3, 3), channels: 192, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_76")
    let max_pooling2d_4 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_4")
    let conv2d_81 = Convolution(kernel: (1, 1), channels: 448, activation: relu, name: "conv2d_81")
    let conv2d_78 = Convolution(kernel: (1, 1), channels: 384, activation: relu, name: "conv2d_78")
    let conv2d_82 = Convolution(kernel: (3, 3), channels: 384, activation: relu, name: "conv2d_82")
    let conv2d_79 = Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "conv2d_79")
    let conv2d_80 = Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "conv2d_80")
    let conv2d_83 = Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "conv2d_83")
    let conv2d_84 = Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "conv2d_84")
    let average_pooling2d_8 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_8")
    let conv2d_77 = Convolution(kernel: (1, 1), channels: 320, activation: relu, name: "conv2d_77")
    let conv2d_85 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_85")
    let conv2d_90 = Convolution(kernel: (1, 1), channels: 448, activation: relu, name: "conv2d_90")
    let conv2d_87 = Convolution(kernel: (1, 1), channels: 384, activation: relu, name: "conv2d_87")
    let conv2d_91 = Convolution(kernel: (3, 3), channels: 384, activation: relu, name: "conv2d_91")
    let conv2d_88 = Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "conv2d_88")
    let conv2d_89 = Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "conv2d_89")
    let conv2d_92 = Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "conv2d_92")
    let conv2d_93 = Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "conv2d_93")
    let average_pooling2d_9 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_9")
    let conv2d_86 = Convolution(kernel: (1, 1), channels: 320, activation: relu, name: "conv2d_86")
    let conv2d_94 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_94")
    let avg_pool = GlobalAveragePooling(name: "avg_pool", useBias: false)
    let predictions = Dense(neurons: 1000, name: "predictions")
    
    do {
      let max_pooling2d_2 = input_2 --> conv2d_1 --> conv2d_2 --> conv2d_3 --> max_pooling2d_1 --> conv2d_4
        --> conv2d_5 --> max_pooling2d_2
      let conv2d_11 = max_pooling2d_2 --> conv2d_9 --> conv2d_10 --> conv2d_11
      let conv2d_6 = max_pooling2d_2 --> conv2d_6
      let conv2d_8 = max_pooling2d_2 --> conv2d_7 --> conv2d_8
      let conv2d_12 = max_pooling2d_2 --> average_pooling2d_1 --> conv2d_12
      let mixed0 = Concatenate([conv2d_6, conv2d_8, conv2d_11, conv2d_12], name: "mixed0")
      let conv2d_13 = mixed0 --> conv2d_13
      let conv2d_19 = mixed0 --> average_pooling2d_2 --> conv2d_19
      let conv2d_18 = mixed0 --> conv2d_16 --> conv2d_17 --> conv2d_18
      let conv2d_15 = mixed0 --> conv2d_14 --> conv2d_15
      let mixed1 = Concatenate([conv2d_13, conv2d_15, conv2d_18, conv2d_19], name: "mixed1")
      let conv2d_26 = mixed1 --> average_pooling2d_3 --> conv2d_26
      let conv2d_22 = mixed1 --> conv2d_21 --> conv2d_22
      let conv2d_25 = mixed1 --> conv2d_23 --> conv2d_24 --> conv2d_25
      let conv2d_20 = mixed1 --> conv2d_20
      let mixed2 = Concatenate([conv2d_20, conv2d_22, conv2d_25, conv2d_26], name: "mixed2")
      let conv2d_30 = mixed2 --> conv2d_28 --> conv2d_29 --> conv2d_30
      let conv2d_27 = mixed2 --> conv2d_27
      let max_pooling2d_3 = mixed2 --> max_pooling2d_3
      let mixed3 = Concatenate([conv2d_27, conv2d_30, max_pooling2d_3], name: "mixed3")
      let conv2d_39 = mixed3 --> conv2d_35 --> conv2d_36 --> conv2d_37 --> conv2d_38 --> conv2d_39
      
      let conv2d_34 = mixed3 --> conv2d_32 --> conv2d_33 --> conv2d_34
      let conv2d_31 = mixed3 --> conv2d_31
      let conv2d_40 = mixed3 --> average_pooling2d_4 --> conv2d_40
      let mixed4 = Concatenate([conv2d_31, conv2d_34, conv2d_39, conv2d_40], name: "mixed4")
      let conv2d_41 = mixed4 --> conv2d_41
      let conv2d_50 = mixed4 --> average_pooling2d_5 --> conv2d_50
      let conv2d_44 = mixed4 --> conv2d_42 --> conv2d_43 --> conv2d_44
      let conv2d_49 = mixed4 --> conv2d_45 --> conv2d_46 --> conv2d_47 --> conv2d_48 --> conv2d_49
      
      let mixed5 = Concatenate([conv2d_41, conv2d_44, conv2d_49, conv2d_50], name: "mixed5")
      let conv2d_51 = mixed5 --> conv2d_51
      let conv2d_59 = mixed5 --> conv2d_55 --> conv2d_56 --> conv2d_57 --> conv2d_58 --> conv2d_59
      
      let conv2d_54 = mixed5 --> conv2d_52 --> conv2d_53 --> conv2d_54
      let conv2d_60 = mixed5 --> average_pooling2d_6 --> conv2d_60
      let mixed6 = Concatenate([conv2d_51, conv2d_54, conv2d_59, conv2d_60], name: "mixed6")
      let conv2d_69 = mixed6 --> conv2d_65 --> conv2d_66 --> conv2d_67 --> conv2d_68 --> conv2d_69
      
      let conv2d_61 = mixed6 --> conv2d_61
      let conv2d_64 = mixed6 --> conv2d_62 --> conv2d_63 --> conv2d_64
      let conv2d_70 = mixed6 --> average_pooling2d_7 --> conv2d_70
      let mixed7 = Concatenate([conv2d_61, conv2d_64, conv2d_69, conv2d_70], name: "mixed7")
      let conv2d_76 = mixed7 --> conv2d_73 --> conv2d_74 --> conv2d_75 --> conv2d_76
      let conv2d_72 = mixed7 --> conv2d_71 --> conv2d_72
      let max_pooling2d_4 = mixed7 --> max_pooling2d_4
      let mixed8 = Concatenate([conv2d_72, conv2d_76, max_pooling2d_4], name: "mixed8")
      let conv2d_82 = mixed8 --> conv2d_81 --> conv2d_82
      let conv2d_77 = mixed8 --> conv2d_77
      let conv2d_78 = mixed8 --> conv2d_78
      let conv2d_84 = conv2d_82 --> conv2d_84
      let conv2d_85 = mixed8 --> average_pooling2d_8 --> conv2d_85
      let conv2d_83 = conv2d_82 --> conv2d_83
      let conv2d_79 = conv2d_78 --> conv2d_79
      let conv2d_80 = conv2d_78 --> conv2d_80
      let mixed9 = Concatenate([conv2d_77, conv2d_79, conv2d_80, conv2d_83, conv2d_84, conv2d_85], name: "mixed9")
      let conv2d_91 = mixed9 --> conv2d_90 --> conv2d_91
      let conv2d_87 = mixed9 --> conv2d_87
      let conv2d_94 = mixed9 --> average_pooling2d_9 --> conv2d_94
      let conv2d_86 = mixed9 --> conv2d_86
      let conv2d_88 = conv2d_87 --> conv2d_88
      let conv2d_93 = conv2d_91 --> conv2d_93
      let conv2d_92 = conv2d_91 --> conv2d_92
      let conv2d_89 = conv2d_87 --> conv2d_89
      let mixed10 = Concatenate([conv2d_86, conv2d_88, conv2d_89, conv2d_92, conv2d_93, conv2d_94], name: "mixed10")
      let predictions = mixed10 --> avg_pool --> predictions
      let output = predictions --> Softmax()
      model = Model(input: input, output: output)
    }
    // end of autogenerated forge net generation code
    var success = false
    let inflightBuffers = 1
    
    if debug {
      success = model.compile(device: device, inflightBuffers: inflightBuffers) {
        name, count, type in ParameterLoaderBundle(name: name,
                                                   count: count,
                                                   prefix: "inception_v3-",
                                                   suffix: type == .weights ? ".weights" : ".biases",
                                                   ext: "bin")
      }
      
    } else {
      success = model.compile(device: device, inflightBuffers: 1) {
        name, count, type in ParameterLoaderRandom(count: count)
      }
    }
    // end of autogenerated forge net generation code
    
    if success {
      print(model.summary())
    }
    
    if debug {
      let testInputImage = loadTexture(named: "final1_299.jpg")
      //let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
      let dumper = DocumentDirectoryDumper(filePrefix: "inception_v3")
      model.debugEvaluate(commandQueue: commandQueue, device: device, input: nil, inputTexure: testInputImage!, dumper: dumper)
      if !dumper.shapes.saveAsJson(fileName: "shapes.json") {
        print("Error dumping shapes")
      }
      
      let probabilitiesImage = model.outputImage(inflightIndex: 0)
      let probabilities = probabilitiesImage.toFloatArray()
      assert(probabilities.count == 1000)
      print("probabilities: \(probabilitiesImage.toFloatArrayChannelsInterleaved())")
      
      typealias Prediction = (label: String, probability: Float, index: Int)
      var result = NeuralNetworkResult<Prediction>()
      result.predictions = probabilities.top(k: 5).map { x -> Prediction in (imageNetLabels[x.0], x.1, x.0) }
      print("predictions:\(result.predictions)")
    } else {
      let test_input_data_3x2_3c_1i : [[[[Float]]]] = [[[[11,12,13],
                                                         [14,15,16]],
                                                        [[21,22,23],
                                                         [24,25,26]],
                                                        [[31,32,33],
                                                         [34,35,36]]]]
      
      let testInputImage = MPSImage(device: device, images: test_input_data_3x2_3c_1i)
      let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
    }
  }
  
  func testInceptionResnet(debug: Bool) {
    print("\(self).\(#function)")
    
    var model : Model
    // begin of autogenerated forge net generation code
    
    let relu = MPSCNNNeuronReLU(device: device, a: 0)
    let input_scale = MPSCNNNeuronLinear(device: device, a: 2, b: -1)
    let input = Input()
    let input_2 = input --> Resize(width: 299, height: 299) --> Activation(input_scale, name: "input_scale")
    let conv2d_1 = Convolution(kernel: (3, 3), channels: 32, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_1")
    let conv2d_2 = Convolution(kernel: (3, 3), channels: 32, padding: .valid, activation: relu, name: "conv2d_2")
    let conv2d_3 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_3")
    let max_pooling2d_1 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_1")
    let conv2d_4 = Convolution(kernel: (1, 1), channels: 80, padding: .valid, activation: relu, name: "conv2d_4")
    let conv2d_5 = Convolution(kernel: (3, 3), channels: 192, padding: .valid, activation: relu, name: "conv2d_5")
    let max_pooling2d_2 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_2")
    let conv2d_9 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_9")
    let conv2d_7 = Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "conv2d_7")
    let conv2d_10 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_10")
    let average_pooling2d_1 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_1")
    let conv2d_6 = Convolution(kernel: (1, 1), channels: 96, activation: relu, name: "conv2d_6")
    let conv2d_8 = Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "conv2d_8")
    let conv2d_11 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_11")
    let conv2d_12 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_12")
    let conv2d_16 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_16")
    let conv2d_14 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_14")
    let conv2d_17 = Convolution(kernel: (3, 3), channels: 48, activation: relu, name: "conv2d_17")
    let conv2d_13 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_13")
    let conv2d_15 = Convolution(kernel: (3, 3), channels: 32, activation: relu, name: "conv2d_15")
    let conv2d_18 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_18")
    let block35_1_conv = Convolution(kernel: (1, 1), channels: 320, name: "block35_1_conv")
    let block35_1_ac = Activation(relu, name: "block35_1_ac")
    let conv2d_22 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_22")
    let conv2d_20 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_20")
    let conv2d_23 = Convolution(kernel: (3, 3), channels: 48, activation: relu, name: "conv2d_23")
    let conv2d_19 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_19")
    let conv2d_21 = Convolution(kernel: (3, 3), channels: 32, activation: relu, name: "conv2d_21")
    let conv2d_24 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_24")
    let block35_2_conv = Convolution(kernel: (1, 1), channels: 320, name: "block35_2_conv")
    let block35_2_ac = Activation(relu, name: "block35_2_ac")
    let conv2d_28 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_28")
    let conv2d_26 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_26")
    let conv2d_29 = Convolution(kernel: (3, 3), channels: 48, activation: relu, name: "conv2d_29")
    let conv2d_25 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_25")
    let conv2d_27 = Convolution(kernel: (3, 3), channels: 32, activation: relu, name: "conv2d_27")
    let conv2d_30 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_30")
    let block35_3_conv = Convolution(kernel: (1, 1), channels: 320, name: "block35_3_conv")
    let block35_3_ac = Activation(relu, name: "block35_3_ac")
    let conv2d_34 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_34")
    let conv2d_32 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_32")
    let conv2d_35 = Convolution(kernel: (3, 3), channels: 48, activation: relu, name: "conv2d_35")
    let conv2d_31 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_31")
    let conv2d_33 = Convolution(kernel: (3, 3), channels: 32, activation: relu, name: "conv2d_33")
    let conv2d_36 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_36")
    let block35_4_conv = Convolution(kernel: (1, 1), channels: 320, name: "block35_4_conv")
    let block35_4_ac = Activation(relu, name: "block35_4_ac")
    let conv2d_40 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_40")
    let conv2d_38 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_38")
    let conv2d_41 = Convolution(kernel: (3, 3), channels: 48, activation: relu, name: "conv2d_41")
    let conv2d_37 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_37")
    let conv2d_39 = Convolution(kernel: (3, 3), channels: 32, activation: relu, name: "conv2d_39")
    let conv2d_42 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_42")
    let block35_5_conv = Convolution(kernel: (1, 1), channels: 320, name: "block35_5_conv")
    let block35_5_ac = Activation(relu, name: "block35_5_ac")
    let conv2d_46 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_46")
    let conv2d_44 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_44")
    let conv2d_47 = Convolution(kernel: (3, 3), channels: 48, activation: relu, name: "conv2d_47")
    let conv2d_43 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_43")
    let conv2d_45 = Convolution(kernel: (3, 3), channels: 32, activation: relu, name: "conv2d_45")
    let conv2d_48 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_48")
    let block35_6_conv = Convolution(kernel: (1, 1), channels: 320, name: "block35_6_conv")
    let block35_6_ac = Activation(relu, name: "block35_6_ac")
    let conv2d_52 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_52")
    let conv2d_50 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_50")
    let conv2d_53 = Convolution(kernel: (3, 3), channels: 48, activation: relu, name: "conv2d_53")
    let conv2d_49 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_49")
    let conv2d_51 = Convolution(kernel: (3, 3), channels: 32, activation: relu, name: "conv2d_51")
    let conv2d_54 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_54")
    let block35_7_conv = Convolution(kernel: (1, 1), channels: 320, name: "block35_7_conv")
    let block35_7_ac = Activation(relu, name: "block35_7_ac")
    let conv2d_58 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_58")
    let conv2d_56 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_56")
    let conv2d_59 = Convolution(kernel: (3, 3), channels: 48, activation: relu, name: "conv2d_59")
    let conv2d_55 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_55")
    let conv2d_57 = Convolution(kernel: (3, 3), channels: 32, activation: relu, name: "conv2d_57")
    let conv2d_60 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_60")
    let block35_8_conv = Convolution(kernel: (1, 1), channels: 320, name: "block35_8_conv")
    let block35_8_ac = Activation(relu, name: "block35_8_ac")
    let conv2d_64 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_64")
    let conv2d_62 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_62")
    let conv2d_65 = Convolution(kernel: (3, 3), channels: 48, activation: relu, name: "conv2d_65")
    let conv2d_61 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_61")
    let conv2d_63 = Convolution(kernel: (3, 3), channels: 32, activation: relu, name: "conv2d_63")
    let conv2d_66 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_66")
    let block35_9_conv = Convolution(kernel: (1, 1), channels: 320, name: "block35_9_conv")
    let block35_9_ac = Activation(relu, name: "block35_9_ac")
    let conv2d_70 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_70")
    let conv2d_68 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_68")
    let conv2d_71 = Convolution(kernel: (3, 3), channels: 48, activation: relu, name: "conv2d_71")
    let conv2d_67 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_67")
    let conv2d_69 = Convolution(kernel: (3, 3), channels: 32, activation: relu, name: "conv2d_69")
    let conv2d_72 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_72")
    let block35_10_conv = Convolution(kernel: (1, 1), channels: 320, name: "block35_10_conv")
    let block35_10_ac = Activation(relu, name: "block35_10_ac")
    let conv2d_74 = Convolution(kernel: (1, 1), channels: 256, activation: relu, name: "conv2d_74")
    let conv2d_75 = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "conv2d_75")
    let conv2d_73 = Convolution(kernel: (3, 3), channels: 384, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_73")
    let conv2d_76 = Convolution(kernel: (3, 3), channels: 384, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_76")
    let max_pooling2d_3 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_3")
    let conv2d_78 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_78")
    let conv2d_79 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_79")
    let conv2d_77 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_77")
    let conv2d_80 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_80")
    let block17_1_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_1_conv")
    let block17_1_ac = Activation(relu, name: "block17_1_ac")
    let conv2d_82 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_82")
    let conv2d_83 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_83")
    let conv2d_81 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_81")
    let conv2d_84 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_84")
    let block17_2_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_2_conv")
    let block17_2_ac = Activation(relu, name: "block17_2_ac")
    let conv2d_86 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_86")
    let conv2d_87 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_87")
    let conv2d_85 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_85")
    let conv2d_88 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_88")
    let block17_3_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_3_conv")
    let block17_3_ac = Activation(relu, name: "block17_3_ac")
    let conv2d_90 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_90")
    let conv2d_91 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_91")
    let conv2d_89 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_89")
    let conv2d_92 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_92")
    let block17_4_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_4_conv")
    let block17_4_ac = Activation(relu, name: "block17_4_ac")
    let conv2d_94 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_94")
    let conv2d_95 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_95")
    let conv2d_93 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_93")
    let conv2d_96 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_96")
    let block17_5_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_5_conv")
    let block17_5_ac = Activation(relu, name: "block17_5_ac")
    let conv2d_98 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_98")
    let conv2d_99 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_99")
    let conv2d_97 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_97")
    let conv2d_100 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_100")
    let block17_6_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_6_conv")
    let block17_6_ac = Activation(relu, name: "block17_6_ac")
    let conv2d_102 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_102")
    let conv2d_103 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_103")
    let conv2d_101 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_101")
    let conv2d_104 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_104")
    let block17_7_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_7_conv")
    let block17_7_ac = Activation(relu, name: "block17_7_ac")
    let conv2d_106 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_106")
    let conv2d_107 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_107")
    let conv2d_105 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_105")
    let conv2d_108 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_108")
    let block17_8_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_8_conv")
    let block17_8_ac = Activation(relu, name: "block17_8_ac")
    let conv2d_110 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_110")
    let conv2d_111 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_111")
    let conv2d_109 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_109")
    let conv2d_112 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_112")
    let block17_9_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_9_conv")
    let block17_9_ac = Activation(relu, name: "block17_9_ac")
    let conv2d_114 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_114")
    let conv2d_115 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_115")
    let conv2d_113 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_113")
    let conv2d_116 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_116")
    let block17_10_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_10_conv")
    let block17_10_ac = Activation(relu, name: "block17_10_ac")
    let conv2d_118 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_118")
    let conv2d_119 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_119")
    let conv2d_117 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_117")
    let conv2d_120 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_120")
    let block17_11_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_11_conv")
    let block17_11_ac = Activation(relu, name: "block17_11_ac")
    let conv2d_122 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_122")
    let conv2d_123 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_123")
    let conv2d_121 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_121")
    let conv2d_124 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_124")
    let block17_12_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_12_conv")
    let block17_12_ac = Activation(relu, name: "block17_12_ac")
    let conv2d_126 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_126")
    let conv2d_127 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_127")
    let conv2d_125 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_125")
    let conv2d_128 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_128")
    let block17_13_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_13_conv")
    let block17_13_ac = Activation(relu, name: "block17_13_ac")
    let conv2d_130 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_130")
    let conv2d_131 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_131")
    let conv2d_129 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_129")
    let conv2d_132 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_132")
    let block17_14_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_14_conv")
    let block17_14_ac = Activation(relu, name: "block17_14_ac")
    let conv2d_134 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_134")
    let conv2d_135 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_135")
    let conv2d_133 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_133")
    let conv2d_136 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_136")
    let block17_15_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_15_conv")
    let block17_15_ac = Activation(relu, name: "block17_15_ac")
    let conv2d_138 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_138")
    let conv2d_139 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_139")
    let conv2d_137 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_137")
    let conv2d_140 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_140")
    let block17_16_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_16_conv")
    let block17_16_ac = Activation(relu, name: "block17_16_ac")
    let conv2d_142 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_142")
    let conv2d_143 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_143")
    let conv2d_141 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_141")
    let conv2d_144 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_144")
    let block17_17_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_17_conv")
    let block17_17_ac = Activation(relu, name: "block17_17_ac")
    let conv2d_146 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_146")
    let conv2d_147 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_147")
    let conv2d_145 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_145")
    let conv2d_148 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_148")
    let block17_18_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_18_conv")
    let block17_18_ac = Activation(relu, name: "block17_18_ac")
    let conv2d_150 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_150")
    let conv2d_151 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_151")
    let conv2d_149 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_149")
    let conv2d_152 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_152")
    let block17_19_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_19_conv")
    let block17_19_ac = Activation(relu, name: "block17_19_ac")
    let conv2d_154 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_154")
    let conv2d_155 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_155")
    let conv2d_153 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_153")
    let conv2d_156 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_156")
    let block17_20_conv = Convolution(kernel: (1, 1), channels: 1088, name: "block17_20_conv")
    let block17_20_ac = Activation(relu, name: "block17_20_ac")
    let conv2d_161 = Convolution(kernel: (1, 1), channels: 256, activation: relu, name: "conv2d_161")
    let conv2d_157 = Convolution(kernel: (1, 1), channels: 256, activation: relu, name: "conv2d_157")
    let conv2d_159 = Convolution(kernel: (1, 1), channels: 256, activation: relu, name: "conv2d_159")
    let conv2d_162 = Convolution(kernel: (3, 3), channels: 288, activation: relu, name: "conv2d_162")
    let conv2d_158 = Convolution(kernel: (3, 3), channels: 384, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_158")
    let conv2d_160 = Convolution(kernel: (3, 3), channels: 288, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_160")
    let conv2d_163 = Convolution(kernel: (3, 3), channels: 320, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_163")
    let max_pooling2d_4 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_4")
    let conv2d_165 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_165")
    let conv2d_166 = Convolution(kernel: (3, 1), channels: 224, activation: relu, name: "conv2d_166")
    let conv2d_164 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_164")
    let conv2d_167 = Convolution(kernel: (1, 3), channels: 256, activation: relu, name: "conv2d_167")
    let block8_1_conv = Convolution(kernel: (1, 1), channels: 2080, name: "block8_1_conv")
    let block8_1_ac = Activation(relu, name: "block8_1_ac")
    let conv2d_169 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_169")
    let conv2d_170 = Convolution(kernel: (3, 1), channels: 224, activation: relu, name: "conv2d_170")
    let conv2d_168 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_168")
    let conv2d_171 = Convolution(kernel: (1, 3), channels: 256, activation: relu, name: "conv2d_171")
    let block8_2_conv = Convolution(kernel: (1, 1), channels: 2080, name: "block8_2_conv")
    let block8_2_ac = Activation(relu, name: "block8_2_ac")
    let conv2d_173 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_173")
    let conv2d_174 = Convolution(kernel: (3, 1), channels: 224, activation: relu, name: "conv2d_174")
    let conv2d_172 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_172")
    let conv2d_175 = Convolution(kernel: (1, 3), channels: 256, activation: relu, name: "conv2d_175")
    let block8_3_conv = Convolution(kernel: (1, 1), channels: 2080, name: "block8_3_conv")
    let block8_3_ac = Activation(relu, name: "block8_3_ac")
    let conv2d_177 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_177")
    let conv2d_178 = Convolution(kernel: (3, 1), channels: 224, activation: relu, name: "conv2d_178")
    let conv2d_176 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_176")
    let conv2d_179 = Convolution(kernel: (1, 3), channels: 256, activation: relu, name: "conv2d_179")
    let block8_4_conv = Convolution(kernel: (1, 1), channels: 2080, name: "block8_4_conv")
    let block8_4_ac = Activation(relu, name: "block8_4_ac")
    let conv2d_181 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_181")
    let conv2d_182 = Convolution(kernel: (3, 1), channels: 224, activation: relu, name: "conv2d_182")
    let conv2d_180 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_180")
    let conv2d_183 = Convolution(kernel: (1, 3), channels: 256, activation: relu, name: "conv2d_183")
    let block8_5_conv = Convolution(kernel: (1, 1), channels: 2080, name: "block8_5_conv")
    let block8_5_ac = Activation(relu, name: "block8_5_ac")
    let conv2d_185 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_185")
    let conv2d_186 = Convolution(kernel: (3, 1), channels: 224, activation: relu, name: "conv2d_186")
    let conv2d_184 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_184")
    let conv2d_187 = Convolution(kernel: (1, 3), channels: 256, activation: relu, name: "conv2d_187")
    let block8_6_conv = Convolution(kernel: (1, 1), channels: 2080, name: "block8_6_conv")
    let block8_6_ac = Activation(relu, name: "block8_6_ac")
    let conv2d_189 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_189")
    let conv2d_190 = Convolution(kernel: (3, 1), channels: 224, activation: relu, name: "conv2d_190")
    let conv2d_188 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_188")
    let conv2d_191 = Convolution(kernel: (1, 3), channels: 256, activation: relu, name: "conv2d_191")
    let block8_7_conv = Convolution(kernel: (1, 1), channels: 2080, name: "block8_7_conv")
    let block8_7_ac = Activation(relu, name: "block8_7_ac")
    let conv2d_193 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_193")
    let conv2d_194 = Convolution(kernel: (3, 1), channels: 224, activation: relu, name: "conv2d_194")
    let conv2d_192 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_192")
    let conv2d_195 = Convolution(kernel: (1, 3), channels: 256, activation: relu, name: "conv2d_195")
    let block8_8_conv = Convolution(kernel: (1, 1), channels: 2080, name: "block8_8_conv")
    let block8_8_ac = Activation(relu, name: "block8_8_ac")
    let conv2d_197 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_197")
    let conv2d_198 = Convolution(kernel: (3, 1), channels: 224, activation: relu, name: "conv2d_198")
    let conv2d_196 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_196")
    let conv2d_199 = Convolution(kernel: (1, 3), channels: 256, activation: relu, name: "conv2d_199")
    let block8_9_conv = Convolution(kernel: (1, 1), channels: 2080, name: "block8_9_conv")
    let block8_9_ac = Activation(relu, name: "block8_9_ac")
    let conv2d_201 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_201")
    let conv2d_202 = Convolution(kernel: (3, 1), channels: 224, activation: relu, name: "conv2d_202")
    let conv2d_200 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_200")
    let conv2d_203 = Convolution(kernel: (1, 3), channels: 256, activation: relu, name: "conv2d_203")
    let block8_10_conv = Convolution(kernel: (1, 1), channels: 2080, name: "block8_10_conv")
    let conv_7b = Convolution(kernel: (1, 1), channels: 1536, activation: relu, name: "conv_7b")
    let avg_pool = GlobalAveragePooling(name: "avg_pool", useBias: false)
    let predictions = Dense(neurons: 1000, name: "predictions")
    
    do {
      let max_pooling2d_2 = input_2 --> conv2d_1 --> conv2d_2 --> conv2d_3 --> max_pooling2d_1 --> conv2d_4
        --> conv2d_5 --> max_pooling2d_2
      let conv2d_6 = max_pooling2d_2 --> conv2d_6
      let conv2d_12 = max_pooling2d_2 --> average_pooling2d_1 --> conv2d_12
      let conv2d_11 = max_pooling2d_2 --> conv2d_9 --> conv2d_10 --> conv2d_11
      let conv2d_8 = max_pooling2d_2 --> conv2d_7 --> conv2d_8
      let mixed_5b = Concatenate([conv2d_6, conv2d_8, conv2d_11, conv2d_12], name: "mixed_5b")
      let conv2d_13 = mixed_5b --> conv2d_13
      let conv2d_15 = mixed_5b --> conv2d_14 --> conv2d_15
      let conv2d_18 = mixed_5b --> conv2d_16 --> conv2d_17 --> conv2d_18
      let block35_1_mixed = Concatenate([conv2d_13, conv2d_15, conv2d_18], name: "block35_1_mixed")
      let block35_1_conv = block35_1_mixed --> block35_1_conv
      let block35_1 = Collect([mixed_5b, block35_1_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.17, b: 0), name: "scale_input1_block35_1")], name: "for_block35_1") --> Add(name: "block35_1")
      let block35_1_ac = block35_1 --> block35_1_ac
      let conv2d_19 = block35_1_ac --> conv2d_19
      let conv2d_21 = block35_1_ac --> conv2d_20 --> conv2d_21
      let conv2d_24 = block35_1_ac --> conv2d_22 --> conv2d_23 --> conv2d_24
      let block35_2_mixed = Concatenate([conv2d_19, conv2d_21, conv2d_24], name: "block35_2_mixed")
      let block35_2_conv = block35_2_mixed --> block35_2_conv
      let block35_2 = Collect([block35_1_ac, block35_2_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.17, b: 0), name: "scale_input1_block35_2")], name: "for_block35_2") --> Add(name: "block35_2")
      let block35_2_ac = block35_2 --> block35_2_ac
      let conv2d_30 = block35_2_ac --> conv2d_28 --> conv2d_29 --> conv2d_30
      let conv2d_25 = block35_2_ac --> conv2d_25
      let conv2d_27 = block35_2_ac --> conv2d_26 --> conv2d_27
      let block35_3_mixed = Concatenate([conv2d_25, conv2d_27, conv2d_30], name: "block35_3_mixed")
      let block35_3_conv = block35_3_mixed --> block35_3_conv
      let block35_3 = Collect([block35_2_ac, block35_3_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.17, b: 0), name: "scale_input1_block35_3")], name: "for_block35_3") --> Add(name: "block35_3")
      let block35_3_ac = block35_3 --> block35_3_ac
      let conv2d_33 = block35_3_ac --> conv2d_32 --> conv2d_33
      let conv2d_31 = block35_3_ac --> conv2d_31
      let conv2d_36 = block35_3_ac --> conv2d_34 --> conv2d_35 --> conv2d_36
      let block35_4_mixed = Concatenate([conv2d_31, conv2d_33, conv2d_36], name: "block35_4_mixed")
      let block35_4_conv = block35_4_mixed --> block35_4_conv
      let block35_4 = Collect([block35_3_ac, block35_4_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.17, b: 0), name: "scale_input1_block35_4")], name: "for_block35_4") --> Add(name: "block35_4")
      let block35_4_ac = block35_4 --> block35_4_ac
      let conv2d_39 = block35_4_ac --> conv2d_38 --> conv2d_39
      let conv2d_42 = block35_4_ac --> conv2d_40 --> conv2d_41 --> conv2d_42
      let conv2d_37 = block35_4_ac --> conv2d_37
      let block35_5_mixed = Concatenate([conv2d_37, conv2d_39, conv2d_42], name: "block35_5_mixed")
      let block35_5_conv = block35_5_mixed --> block35_5_conv
      let block35_5 = Collect([block35_4_ac, block35_5_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.17, b: 0), name: "scale_input1_block35_5")], name: "for_block35_5") --> Add(name: "block35_5")
      let block35_5_ac = block35_5 --> block35_5_ac
      let conv2d_43 = block35_5_ac --> conv2d_43
      let conv2d_48 = block35_5_ac --> conv2d_46 --> conv2d_47 --> conv2d_48
      let conv2d_45 = block35_5_ac --> conv2d_44 --> conv2d_45
      let block35_6_mixed = Concatenate([conv2d_43, conv2d_45, conv2d_48], name: "block35_6_mixed")
      let block35_6_conv = block35_6_mixed --> block35_6_conv
      let block35_6 = Collect([block35_5_ac, block35_6_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.17, b: 0), name: "scale_input1_block35_6")], name: "for_block35_6") --> Add(name: "block35_6")
      let block35_6_ac = block35_6 --> block35_6_ac
      let conv2d_49 = block35_6_ac --> conv2d_49
      let conv2d_54 = block35_6_ac --> conv2d_52 --> conv2d_53 --> conv2d_54
      let conv2d_51 = block35_6_ac --> conv2d_50 --> conv2d_51
      let block35_7_mixed = Concatenate([conv2d_49, conv2d_51, conv2d_54], name: "block35_7_mixed")
      let block35_7_conv = block35_7_mixed --> block35_7_conv
      let block35_7 = Collect([block35_6_ac, block35_7_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.17, b: 0), name: "scale_input1_block35_7")], name: "for_block35_7") --> Add(name: "block35_7")
      let block35_7_ac = block35_7 --> block35_7_ac
      let conv2d_57 = block35_7_ac --> conv2d_56 --> conv2d_57
      let conv2d_55 = block35_7_ac --> conv2d_55
      let conv2d_60 = block35_7_ac --> conv2d_58 --> conv2d_59 --> conv2d_60
      let block35_8_mixed = Concatenate([conv2d_55, conv2d_57, conv2d_60], name: "block35_8_mixed")
      let block35_8_conv = block35_8_mixed --> block35_8_conv
      let block35_8 = Collect([block35_7_ac, block35_8_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.17, b: 0), name: "scale_input1_block35_8")], name: "for_block35_8") --> Add(name: "block35_8")
      let block35_8_ac = block35_8 --> block35_8_ac
      let conv2d_63 = block35_8_ac --> conv2d_62 --> conv2d_63
      let conv2d_61 = block35_8_ac --> conv2d_61
      let conv2d_66 = block35_8_ac --> conv2d_64 --> conv2d_65 --> conv2d_66
      let block35_9_mixed = Concatenate([conv2d_61, conv2d_63, conv2d_66], name: "block35_9_mixed")
      let block35_9_conv = block35_9_mixed --> block35_9_conv
      let block35_9 = Collect([block35_8_ac, block35_9_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.17, b: 0), name: "scale_input1_block35_9")], name: "for_block35_9") --> Add(name: "block35_9")
      let block35_9_ac = block35_9 --> block35_9_ac
      let conv2d_67 = block35_9_ac --> conv2d_67
      let conv2d_69 = block35_9_ac --> conv2d_68 --> conv2d_69
      let conv2d_72 = block35_9_ac --> conv2d_70 --> conv2d_71 --> conv2d_72
      let block35_10_mixed = Concatenate([conv2d_67, conv2d_69, conv2d_72], name: "block35_10_mixed")
      let block35_10_conv = block35_10_mixed --> block35_10_conv
      let block35_10 = Collect([block35_9_ac, block35_10_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.17, b: 0), name: "scale_input1_block35_10")], name: "for_block35_10") --> Add(name: "block35_10")
      let block35_10_ac = block35_10 --> block35_10_ac
      let conv2d_73 = block35_10_ac --> conv2d_73
      let max_pooling2d_3 = block35_10_ac --> max_pooling2d_3
      let conv2d_76 = block35_10_ac --> conv2d_74 --> conv2d_75 --> conv2d_76
      let mixed_6a = Concatenate([conv2d_73, conv2d_76, max_pooling2d_3], name: "mixed_6a")
      let conv2d_80 = mixed_6a --> conv2d_78 --> conv2d_79 --> conv2d_80
      let conv2d_77 = mixed_6a --> conv2d_77
      let block17_1_mixed = Concatenate([conv2d_77, conv2d_80], name: "block17_1_mixed")
      let block17_1_conv = block17_1_mixed --> block17_1_conv
      let block17_1 = Collect([mixed_6a, block17_1_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_1")], name: "for_block17_1") --> Add(name: "block17_1")
      let block17_1_ac = block17_1 --> block17_1_ac
      let conv2d_81 = block17_1_ac --> conv2d_81
      let conv2d_84 = block17_1_ac --> conv2d_82 --> conv2d_83 --> conv2d_84
      let block17_2_mixed = Concatenate([conv2d_81, conv2d_84], name: "block17_2_mixed")
      let block17_2_conv = block17_2_mixed --> block17_2_conv
      let block17_2 = Collect([block17_1_ac, block17_2_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_2")], name: "for_block17_2") --> Add(name: "block17_2")
      let block17_2_ac = block17_2 --> block17_2_ac
      let conv2d_88 = block17_2_ac --> conv2d_86 --> conv2d_87 --> conv2d_88
      let conv2d_85 = block17_2_ac --> conv2d_85
      let block17_3_mixed = Concatenate([conv2d_85, conv2d_88], name: "block17_3_mixed")
      let block17_3_conv = block17_3_mixed --> block17_3_conv
      let block17_3 = Collect([block17_2_ac, block17_3_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_3")], name: "for_block17_3") --> Add(name: "block17_3")
      let block17_3_ac = block17_3 --> block17_3_ac
      let conv2d_89 = block17_3_ac --> conv2d_89
      let conv2d_92 = block17_3_ac --> conv2d_90 --> conv2d_91 --> conv2d_92
      let block17_4_mixed = Concatenate([conv2d_89, conv2d_92], name: "block17_4_mixed")
      let block17_4_conv = block17_4_mixed --> block17_4_conv
      let block17_4 = Collect([block17_3_ac, block17_4_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_4")], name: "for_block17_4") --> Add(name: "block17_4")
      let block17_4_ac = block17_4 --> block17_4_ac
      let conv2d_96 = block17_4_ac --> conv2d_94 --> conv2d_95 --> conv2d_96
      let conv2d_93 = block17_4_ac --> conv2d_93
      let block17_5_mixed = Concatenate([conv2d_93, conv2d_96], name: "block17_5_mixed")
      let block17_5_conv = block17_5_mixed --> block17_5_conv
      let block17_5 = Collect([block17_4_ac, block17_5_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_5")], name: "for_block17_5") --> Add(name: "block17_5")
      let block17_5_ac = block17_5 --> block17_5_ac
      let conv2d_100 = block17_5_ac --> conv2d_98 --> conv2d_99 --> conv2d_100
      let conv2d_97 = block17_5_ac --> conv2d_97
      let block17_6_mixed = Concatenate([conv2d_97, conv2d_100], name: "block17_6_mixed")
      let block17_6_conv = block17_6_mixed --> block17_6_conv
      let block17_6 = Collect([block17_5_ac, block17_6_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_6")], name: "for_block17_6") --> Add(name: "block17_6")
      let block17_6_ac = block17_6 --> block17_6_ac
      let conv2d_101 = block17_6_ac --> conv2d_101
      let conv2d_104 = block17_6_ac --> conv2d_102 --> conv2d_103 --> conv2d_104
      let block17_7_mixed = Concatenate([conv2d_101, conv2d_104], name: "block17_7_mixed")
      let block17_7_conv = block17_7_mixed --> block17_7_conv
      let block17_7 = Collect([block17_6_ac, block17_7_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_7")], name: "for_block17_7") --> Add(name: "block17_7")
      let block17_7_ac = block17_7 --> block17_7_ac
      let conv2d_108 = block17_7_ac --> conv2d_106 --> conv2d_107 --> conv2d_108
      let conv2d_105 = block17_7_ac --> conv2d_105
      let block17_8_mixed = Concatenate([conv2d_105, conv2d_108], name: "block17_8_mixed")
      let block17_8_conv = block17_8_mixed --> block17_8_conv
      let block17_8 = Collect([block17_7_ac, block17_8_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_8")], name: "for_block17_8") --> Add(name: "block17_8")
      let block17_8_ac = block17_8 --> block17_8_ac
      let conv2d_109 = block17_8_ac --> conv2d_109
      let conv2d_112 = block17_8_ac --> conv2d_110 --> conv2d_111 --> conv2d_112
      let block17_9_mixed = Concatenate([conv2d_109, conv2d_112], name: "block17_9_mixed")
      let block17_9_conv = block17_9_mixed --> block17_9_conv
      let block17_9 = Collect([block17_8_ac, block17_9_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_9")], name: "for_block17_9") --> Add(name: "block17_9")
      let block17_9_ac = block17_9 --> block17_9_ac
      let conv2d_116 = block17_9_ac --> conv2d_114 --> conv2d_115 --> conv2d_116
      let conv2d_113 = block17_9_ac --> conv2d_113
      let block17_10_mixed = Concatenate([conv2d_113, conv2d_116], name: "block17_10_mixed")
      let block17_10_conv = block17_10_mixed --> block17_10_conv
      let block17_10 = Collect([block17_9_ac, block17_10_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_10")], name: "for_block17_10") --> Add(name: "block17_10")
      let block17_10_ac = block17_10 --> block17_10_ac
      let conv2d_117 = block17_10_ac --> conv2d_117
      let conv2d_120 = block17_10_ac --> conv2d_118 --> conv2d_119 --> conv2d_120
      let block17_11_mixed = Concatenate([conv2d_117, conv2d_120], name: "block17_11_mixed")
      let block17_11_conv = block17_11_mixed --> block17_11_conv
      let block17_11 = Collect([block17_10_ac, block17_11_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_11")], name: "for_block17_11") --> Add(name: "block17_11")
      let block17_11_ac = block17_11 --> block17_11_ac
      let conv2d_124 = block17_11_ac --> conv2d_122 --> conv2d_123 --> conv2d_124
      let conv2d_121 = block17_11_ac --> conv2d_121
      let block17_12_mixed = Concatenate([conv2d_121, conv2d_124], name: "block17_12_mixed")
      let block17_12_conv = block17_12_mixed --> block17_12_conv
      let block17_12 = Collect([block17_11_ac, block17_12_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_12")], name: "for_block17_12") --> Add(name: "block17_12")
      let block17_12_ac = block17_12 --> block17_12_ac
      let conv2d_125 = block17_12_ac --> conv2d_125
      let conv2d_128 = block17_12_ac --> conv2d_126 --> conv2d_127 --> conv2d_128
      let block17_13_mixed = Concatenate([conv2d_125, conv2d_128], name: "block17_13_mixed")
      let block17_13_conv = block17_13_mixed --> block17_13_conv
      let block17_13 = Collect([block17_12_ac, block17_13_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_13")], name: "for_block17_13") --> Add(name: "block17_13")
      let block17_13_ac = block17_13 --> block17_13_ac
      let conv2d_129 = block17_13_ac --> conv2d_129
      let conv2d_132 = block17_13_ac --> conv2d_130 --> conv2d_131 --> conv2d_132
      let block17_14_mixed = Concatenate([conv2d_129, conv2d_132], name: "block17_14_mixed")
      let block17_14_conv = block17_14_mixed --> block17_14_conv
      let block17_14 = Collect([block17_13_ac, block17_14_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_14")], name: "for_block17_14") --> Add(name: "block17_14")
      let block17_14_ac = block17_14 --> block17_14_ac
      let conv2d_136 = block17_14_ac --> conv2d_134 --> conv2d_135 --> conv2d_136
      let conv2d_133 = block17_14_ac --> conv2d_133
      let block17_15_mixed = Concatenate([conv2d_133, conv2d_136], name: "block17_15_mixed")
      let block17_15_conv = block17_15_mixed --> block17_15_conv
      let block17_15 = Collect([block17_14_ac, block17_15_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_15")], name: "for_block17_15") --> Add(name: "block17_15")
      let block17_15_ac = block17_15 --> block17_15_ac
      let conv2d_137 = block17_15_ac --> conv2d_137
      let conv2d_140 = block17_15_ac --> conv2d_138 --> conv2d_139 --> conv2d_140
      let block17_16_mixed = Concatenate([conv2d_137, conv2d_140], name: "block17_16_mixed")
      let block17_16_conv = block17_16_mixed --> block17_16_conv
      let block17_16 = Collect([block17_15_ac, block17_16_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_16")], name: "for_block17_16") --> Add(name: "block17_16")
      let block17_16_ac = block17_16 --> block17_16_ac
      let conv2d_144 = block17_16_ac --> conv2d_142 --> conv2d_143 --> conv2d_144
      let conv2d_141 = block17_16_ac --> conv2d_141
      let block17_17_mixed = Concatenate([conv2d_141, conv2d_144], name: "block17_17_mixed")
      let block17_17_conv = block17_17_mixed --> block17_17_conv
      let block17_17 = Collect([block17_16_ac, block17_17_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_17")], name: "for_block17_17") --> Add(name: "block17_17")
      let block17_17_ac = block17_17 --> block17_17_ac
      let conv2d_148 = block17_17_ac --> conv2d_146 --> conv2d_147 --> conv2d_148
      let conv2d_145 = block17_17_ac --> conv2d_145
      let block17_18_mixed = Concatenate([conv2d_145, conv2d_148], name: "block17_18_mixed")
      let block17_18_conv = block17_18_mixed --> block17_18_conv
      let block17_18 = Collect([block17_17_ac, block17_18_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_18")], name: "for_block17_18") --> Add(name: "block17_18")
      let block17_18_ac = block17_18 --> block17_18_ac
      let conv2d_152 = block17_18_ac --> conv2d_150 --> conv2d_151 --> conv2d_152
      let conv2d_149 = block17_18_ac --> conv2d_149
      let block17_19_mixed = Concatenate([conv2d_149, conv2d_152], name: "block17_19_mixed")
      let block17_19_conv = block17_19_mixed --> block17_19_conv
      let block17_19 = Collect([block17_18_ac, block17_19_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_19")], name: "for_block17_19") --> Add(name: "block17_19")
      let block17_19_ac = block17_19 --> block17_19_ac
      let conv2d_153 = block17_19_ac --> conv2d_153
      let conv2d_156 = block17_19_ac --> conv2d_154 --> conv2d_155 --> conv2d_156
      let block17_20_mixed = Concatenate([conv2d_153, conv2d_156], name: "block17_20_mixed")
      let block17_20_conv = block17_20_mixed --> block17_20_conv
      let block17_20 = Collect([block17_19_ac, block17_20_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.1, b: 0), name: "scale_input1_block17_20")], name: "for_block17_20") --> Add(name: "block17_20")
      let block17_20_ac = block17_20 --> block17_20_ac
      let max_pooling2d_4 = block17_20_ac --> max_pooling2d_4
      let conv2d_160 = block17_20_ac --> conv2d_159 --> conv2d_160
      let conv2d_163 = block17_20_ac --> conv2d_161 --> conv2d_162 --> conv2d_163
      let conv2d_158 = block17_20_ac --> conv2d_157 --> conv2d_158
      let mixed_7a = Concatenate([conv2d_158, conv2d_160, conv2d_163, max_pooling2d_4], name: "mixed_7a")
      let conv2d_167 = mixed_7a --> conv2d_165 --> conv2d_166 --> conv2d_167
      let conv2d_164 = mixed_7a --> conv2d_164
      let block8_1_mixed = Concatenate([conv2d_164, conv2d_167], name: "block8_1_mixed")
      let block8_1_conv = block8_1_mixed --> block8_1_conv
      let block8_1 = Collect([mixed_7a, block8_1_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.2, b: 0), name: "scale_input1_block8_1")], name: "for_block8_1") --> Add(name: "block8_1")
      let block8_1_ac = block8_1 --> block8_1_ac
      let conv2d_171 = block8_1_ac --> conv2d_169 --> conv2d_170 --> conv2d_171
      let conv2d_168 = block8_1_ac --> conv2d_168
      let block8_2_mixed = Concatenate([conv2d_168, conv2d_171], name: "block8_2_mixed")
      let block8_2_conv = block8_2_mixed --> block8_2_conv
      let block8_2 = Collect([block8_1_ac, block8_2_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.2, b: 0), name: "scale_input1_block8_2")], name: "for_block8_2") --> Add(name: "block8_2")
      let block8_2_ac = block8_2 --> block8_2_ac
      let conv2d_172 = block8_2_ac --> conv2d_172
      let conv2d_175 = block8_2_ac --> conv2d_173 --> conv2d_174 --> conv2d_175
      let block8_3_mixed = Concatenate([conv2d_172, conv2d_175], name: "block8_3_mixed")
      let block8_3_conv = block8_3_mixed --> block8_3_conv
      let block8_3 = Collect([block8_2_ac, block8_3_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.2, b: 0), name: "scale_input1_block8_3")], name: "for_block8_3") --> Add(name: "block8_3")
      let block8_3_ac = block8_3 --> block8_3_ac
      let conv2d_179 = block8_3_ac --> conv2d_177 --> conv2d_178 --> conv2d_179
      let conv2d_176 = block8_3_ac --> conv2d_176
      let block8_4_mixed = Concatenate([conv2d_176, conv2d_179], name: "block8_4_mixed")
      let block8_4_conv = block8_4_mixed --> block8_4_conv
      let block8_4 = Collect([block8_3_ac, block8_4_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.2, b: 0), name: "scale_input1_block8_4")], name: "for_block8_4") --> Add(name: "block8_4")
      let block8_4_ac = block8_4 --> block8_4_ac
      let conv2d_180 = block8_4_ac --> conv2d_180
      let conv2d_183 = block8_4_ac --> conv2d_181 --> conv2d_182 --> conv2d_183
      let block8_5_mixed = Concatenate([conv2d_180, conv2d_183], name: "block8_5_mixed")
      let block8_5_conv = block8_5_mixed --> block8_5_conv
      let block8_5 = Collect([block8_4_ac, block8_5_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.2, b: 0), name: "scale_input1_block8_5")], name: "for_block8_5") --> Add(name: "block8_5")
      let block8_5_ac = block8_5 --> block8_5_ac
      let conv2d_187 = block8_5_ac --> conv2d_185 --> conv2d_186 --> conv2d_187
      let conv2d_184 = block8_5_ac --> conv2d_184
      let block8_6_mixed = Concatenate([conv2d_184, conv2d_187], name: "block8_6_mixed")
      let block8_6_conv = block8_6_mixed --> block8_6_conv
      let block8_6 = Collect([block8_5_ac, block8_6_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.2, b: 0), name: "scale_input1_block8_6")], name: "for_block8_6") --> Add(name: "block8_6")
      let block8_6_ac = block8_6 --> block8_6_ac
      let conv2d_188 = block8_6_ac --> conv2d_188
      let conv2d_191 = block8_6_ac --> conv2d_189 --> conv2d_190 --> conv2d_191
      let block8_7_mixed = Concatenate([conv2d_188, conv2d_191], name: "block8_7_mixed")
      let block8_7_conv = block8_7_mixed --> block8_7_conv
      let block8_7 = Collect([block8_6_ac, block8_7_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.2, b: 0), name: "scale_input1_block8_7")], name: "for_block8_7") --> Add(name: "block8_7")
      let block8_7_ac = block8_7 --> block8_7_ac
      let conv2d_195 = block8_7_ac --> conv2d_193 --> conv2d_194 --> conv2d_195
      let conv2d_192 = block8_7_ac --> conv2d_192
      let block8_8_mixed = Concatenate([conv2d_192, conv2d_195], name: "block8_8_mixed")
      let block8_8_conv = block8_8_mixed --> block8_8_conv
      let block8_8 = Collect([block8_7_ac, block8_8_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.2, b: 0), name: "scale_input1_block8_8")], name: "for_block8_8") --> Add(name: "block8_8")
      let block8_8_ac = block8_8 --> block8_8_ac
      let conv2d_196 = block8_8_ac --> conv2d_196
      let conv2d_199 = block8_8_ac --> conv2d_197 --> conv2d_198 --> conv2d_199
      let block8_9_mixed = Concatenate([conv2d_196, conv2d_199], name: "block8_9_mixed")
      let block8_9_conv = block8_9_mixed --> block8_9_conv
      let block8_9 = Collect([block8_8_ac, block8_9_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 0.2, b: 0), name: "scale_input1_block8_9")], name: "for_block8_9") --> Add(name: "block8_9")
      let block8_9_ac = block8_9 --> block8_9_ac
      let conv2d_203 = block8_9_ac --> conv2d_201 --> conv2d_202 --> conv2d_203
      let conv2d_200 = block8_9_ac --> conv2d_200
      let block8_10_mixed = Concatenate([conv2d_200, conv2d_203], name: "block8_10_mixed")
      let block8_10_conv = block8_10_mixed --> block8_10_conv
      let block8_10 = Collect([block8_9_ac, block8_10_conv --> Activation(MPSCNNNeuronLinear(device: device, a: 1.0, b: 0), name: "scale_input1_block8_10")], name: "for_block8_10") --> Add(name: "block8_10")
      let predictions = block8_10 --> conv_7b --> avg_pool --> predictions
      let output = predictions --> Softmax()
      model = Model(input: input, output: output)
    }
    
    // end of autogenerated forge net generation code
    var success = false
    let inflightBuffers = 1
    
    if debug {
      success = model.compile(device: device, inflightBuffers: inflightBuffers) {
        name, count, type in ParameterLoaderBundle(name: name,
                                                   count: count,
                                                   prefix: "inception_resnet_v2-",
                                                   suffix: type == .weights ? ".weights" : ".biases",
                                                   ext: "bin")
      }
      
    } else {
      success = model.compile(device: device, inflightBuffers: 1) {
        name, count, type in ParameterLoaderRandom(count: count)
      }
    }
    // end of autogenerated forge net generation code
    
    if success {
      print(model.summary())
    }
    
    if debug {
      let testInputImage = loadTexture(named: "final1_299.jpg")
      //let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
      let dumper = DocumentDirectoryDumper(filePrefix: "inception_resnet_v2")
      model.debugEvaluate(commandQueue: commandQueue, device: device, input: nil, inputTexure: testInputImage!, dumper: dumper)
      //model.debugTrace = true
      //let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: nil, inputTexure: testInputImage!)

      if !dumper.shapes.saveAsJson(fileName: "shapes.json") {
        print("Error dumping shapes")
      }
      
      let probabilitiesImage = model.outputImage(inflightIndex: 0)
      let probabilities = probabilitiesImage.toFloatArray()
      assert(probabilities.count == 1000)
      print("probabilities: \(probabilitiesImage.toFloatArrayChannelsInterleaved())")
      
      typealias Prediction = (label: String, probability: Float, index: Int)
      var result = NeuralNetworkResult<Prediction>()
      result.predictions = probabilities.top(k: 5).map { x -> Prediction in (imageNetLabels[x.0], x.1, x.0) }
      print("predictions:\(result.predictions)")
    } else {
      let test_input_data_3x2_3c_1i : [[[[Float]]]] = [[[[11,12,13],
                                                         [14,15,16]],
                                                        [[21,22,23],
                                                         [24,25,26]],
                                                        [[31,32,33],
                                                         [34,35,36]]]]
      
      let testInputImage = MPSImage(device: device, images: test_input_data_3x2_3c_1i)
      let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
    }
  }

  
  func testInception_V3_old(debug: Bool) {
    print("\(self).\(#function)")
    
    var model : Model
    // begin of autogenerated forge net generation code
    
    let relu = MPSCNNNeuronReLU(device: device, a: 0)
    //let input_scale = MPSCNNNeuronLinear(device: device, a: 1.0 / 128.0, b: 0)
    let input_scale = MPSCNNNeuronLinear(device: device, a: 2, b: -1)
    let input = Input()
    //let swap_channels = Custom(TransposeChannelsKernel(device: device, featureChannels: 3, permute: [2,1,0]), name: "rgb2bgr")
    //let subtract_mean = Custom(SubtractMeanColor(device:device, red: 123.68, green: 116.779, blue: 103.939, scale: 255.0), name: "subtract_mean")
    //let input_2 = input --> Resize(width: 299, height: 299) -->  swap_channels -->  subtract_mean --> Activation(input_scale)
    let input_2 = input --> Resize(width: 299, height: 299) -->  Activation(input_scale)
    let conv2d_1 = Convolution(kernel: (3, 3), channels: 32, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_1")
    let conv2d_2 = Convolution(kernel: (3, 3), channels: 32, padding: .valid, activation: relu, name: "conv2d_2")
    let conv2d_3 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_3")
    let max_pooling2d_1 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_1")
    let conv2d_4 = Convolution(kernel: (1, 1), channels: 80, padding: .valid, activation: relu, name: "conv2d_4")
    let conv2d_5 = Convolution(kernel: (3, 3), channels: 192, padding: .valid, activation: relu, name: "conv2d_5")
    let max_pooling2d_2 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_2")
    let conv2d_9 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_9")
    let conv2d_7 = Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "conv2d_7")
    let conv2d_10 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_10")
    let average_pooling2d_1 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_1")
    let conv2d_6 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_6")
    let conv2d_8 = Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "conv2d_8")
    let conv2d_11 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_11")
    let conv2d_12 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_12")
    let conv2d_16 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_16")
    let conv2d_14 = Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "conv2d_14")
    let conv2d_17 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_17")
    let average_pooling2d_2 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_2")
    let conv2d_13 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_13")
    let conv2d_15 = Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "conv2d_15")
    let conv2d_18 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_18")
    let conv2d_19 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_19")
    let conv2d_23 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_23")
    let conv2d_21 = Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "conv2d_21")
    let conv2d_24 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_24")
    let average_pooling2d_3 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_3")
    let conv2d_20 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_20")
    let conv2d_22 = Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "conv2d_22")
    let conv2d_25 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_25")
    let conv2d_26 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_26")
    let conv2d_28 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_28")
    let conv2d_29 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_29")
    let conv2d_27 = Convolution(kernel: (3, 3), channels: 384, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_27")
    let conv2d_30 = Convolution(kernel: (3, 3), channels: 96, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_30")
    let max_pooling2d_3 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_3")
    let conv2d_35 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_35")
    let conv2d_36 = Convolution(kernel: (7, 1), channels: 128, activation: relu, name: "conv2d_36")
    let conv2d_32 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_32")
    let conv2d_37 = Convolution(kernel: (1, 7), channels: 128, activation: relu, name: "conv2d_37")
    let conv2d_33 = Convolution(kernel: (1, 7), channels: 128, activation: relu, name: "conv2d_33")
    //let conv2d_33 = Convolution(kernel: (7, 1), channels: 128, activation: relu, name: "conv2d_33")
    let conv2d_38 = Convolution(kernel: (7, 1), channels: 128, activation: relu, name: "conv2d_38")
    let average_pooling2d_4 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_4")
    let conv2d_31 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_31")
    let conv2d_34 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_34")
    let conv2d_39 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_39")
    let conv2d_40 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_40")
    let conv2d_45 = Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "conv2d_45")
    let conv2d_46 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_46")
    let conv2d_42 = Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "conv2d_42")
    let conv2d_47 = Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "conv2d_47")
    let conv2d_43 = Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "conv2d_43")
    let conv2d_48 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_48")
    let average_pooling2d_5 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_5")
    let conv2d_41 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_41")
    let conv2d_44 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_44")
    let conv2d_49 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_49")
    let conv2d_50 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_50")
    let conv2d_55 = Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "conv2d_55")
    let conv2d_56 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_56")
    let conv2d_52 = Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "conv2d_52")
    let conv2d_57 = Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "conv2d_57")
    let conv2d_53 = Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "conv2d_53")
    let conv2d_58 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_58")
    let average_pooling2d_6 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_6")
    let conv2d_51 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_51")
    let conv2d_54 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_54")
    let conv2d_59 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_59")
    let conv2d_60 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_60")
    let conv2d_65 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_65")
    let conv2d_66 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_66")
    let conv2d_62 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_62")
    let conv2d_67 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_67")
    let conv2d_63 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_63")
    let conv2d_68 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_68")
    let average_pooling2d_7 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_7")
    let conv2d_61 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_61")
    let conv2d_64 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_64")
    let conv2d_69 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_69")
    let conv2d_70 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_70")
    let conv2d_73 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_73")
    let conv2d_74 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_74")
    let conv2d_71 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_71")
    let conv2d_75 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_75")
    let conv2d_72 = Convolution(kernel: (3, 3), channels: 320, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_72")
    let conv2d_76 = Convolution(kernel: (3, 3), channels: 192, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_76")
    let max_pooling2d_4 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_4")
    let conv2d_81 = Convolution(kernel: (1, 1), channels: 448, activation: relu, name: "conv2d_81")
    let conv2d_78 = Convolution(kernel: (1, 1), channels: 384, activation: relu, name: "conv2d_78")
    let conv2d_82 = Convolution(kernel: (3, 3), channels: 384, activation: relu, name: "conv2d_82")
    let conv2d_79 = Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "conv2d_79")
    let conv2d_80 = Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "conv2d_80")
    let conv2d_83 = Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "conv2d_83")
    let conv2d_84 = Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "conv2d_84")
    let average_pooling2d_8 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_8")
    let conv2d_77 = Convolution(kernel: (1, 1), channels: 320, activation: relu, name: "conv2d_77")
    let conv2d_85 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_85")
    let conv2d_90 = Convolution(kernel: (1, 1), channels: 448, activation: relu, name: "conv2d_90")
    let conv2d_87 = Convolution(kernel: (1, 1), channels: 384, activation: relu, name: "conv2d_87")
    let conv2d_91 = Convolution(kernel: (3, 3), channels: 384, activation: relu, name: "conv2d_91")
    let conv2d_88 = Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "conv2d_88")
    let conv2d_89 = Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "conv2d_89")
    let conv2d_92 = Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "conv2d_92")
    let conv2d_93 = Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "conv2d_93")
    let average_pooling2d_9 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_9")
    let conv2d_86 = Convolution(kernel: (1, 1), channels: 320, activation: relu, name: "conv2d_86")
    let conv2d_94 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_94")
    let avg_pool = GlobalAveragePooling(name: "avg_pool", useBias: false)
    let predictions = Dense(neurons: 1000, name: "predictions")
    
    do {
      let max_pooling2d_2 = input_2 --> conv2d_1 --> conv2d_2 --> conv2d_3 --> max_pooling2d_1 --> conv2d_4
        --> conv2d_5 --> max_pooling2d_2
      let conv2d_6 = max_pooling2d_2 --> conv2d_6
      let conv2d_12 = max_pooling2d_2 --> average_pooling2d_1 --> conv2d_12
      let conv2d_8 = max_pooling2d_2 --> conv2d_7 --> conv2d_8
      let conv2d_11 = max_pooling2d_2 --> conv2d_9 --> conv2d_10 --> conv2d_11
      let mixed0 = Concatenate([conv2d_6, conv2d_8, conv2d_11, conv2d_12], name: "mixed0")
      let conv2d_15 = mixed0 --> conv2d_14 --> conv2d_15
      let conv2d_18 = mixed0 --> conv2d_16 --> conv2d_17 --> conv2d_18
      let conv2d_13 = mixed0 --> conv2d_13
      let conv2d_19 = mixed0 --> average_pooling2d_2 --> conv2d_19
      let mixed1 = Concatenate([conv2d_13, conv2d_15, conv2d_18, conv2d_19], name: "mixed1")
      let conv2d_20 = mixed1 --> conv2d_20
      let conv2d_22 = mixed1 --> conv2d_21 --> conv2d_22
      let conv2d_26 = mixed1 --> average_pooling2d_3 --> conv2d_26
      let conv2d_25 = mixed1 --> conv2d_23 --> conv2d_24 --> conv2d_25
      let mixed2 = Concatenate([conv2d_20, conv2d_22, conv2d_25, conv2d_26], name: "mixed2")
      let max_pooling2d_3 = mixed2 --> max_pooling2d_3
      let conv2d_27 = mixed2 --> conv2d_27
      let conv2d_30 = mixed2 --> conv2d_28 --> conv2d_29 --> conv2d_30
      let mixed3 = Concatenate([conv2d_27, conv2d_30, max_pooling2d_3], name: "mixed3")
      let conv2d_39 = mixed3 --> conv2d_35 --> conv2d_36 --> conv2d_37 --> conv2d_38 --> conv2d_39
      
      let conv2d_31 = mixed3 --> conv2d_31
      let conv2d_40 = mixed3 --> average_pooling2d_4 --> conv2d_40
      let conv2d_34 = mixed3 --> conv2d_32 --> conv2d_33 --> conv2d_34
      let mixed4 = Concatenate([conv2d_31, conv2d_34, conv2d_39, conv2d_40], name: "mixed4")
      let conv2d_50 = mixed4 --> average_pooling2d_5 --> conv2d_50
      let conv2d_44 = mixed4 --> conv2d_42 --> conv2d_43 --> conv2d_44
      let conv2d_49 = mixed4 --> conv2d_45 --> conv2d_46 --> conv2d_47 --> conv2d_48 --> conv2d_49
      
      let conv2d_41 = mixed4 --> conv2d_41
      let mixed5 = Concatenate([conv2d_41, conv2d_44, conv2d_49, conv2d_50], name: "mixed5")
      let conv2d_59 = mixed5 --> conv2d_55 --> conv2d_56 --> conv2d_57 --> conv2d_58 --> conv2d_59
      
      let conv2d_51 = mixed5 --> conv2d_51
      let conv2d_54 = mixed5 --> conv2d_52 --> conv2d_53 --> conv2d_54
      let conv2d_60 = mixed5 --> average_pooling2d_6 --> conv2d_60
      let mixed6 = Concatenate([conv2d_51, conv2d_54, conv2d_59, conv2d_60], name: "mixed6")
      let conv2d_69 = mixed6 --> conv2d_65 --> conv2d_66 --> conv2d_67 --> conv2d_68 --> conv2d_69
      
      let conv2d_61 = mixed6 --> conv2d_61
      let conv2d_64 = mixed6 --> conv2d_62 --> conv2d_63 --> conv2d_64
      let conv2d_70 = mixed6 --> average_pooling2d_7 --> conv2d_70
      let mixed7 = Concatenate([conv2d_61, conv2d_64, conv2d_69, conv2d_70], name: "mixed7")
      let conv2d_76 = mixed7 --> conv2d_73 --> conv2d_74 --> conv2d_75 --> conv2d_76
      let max_pooling2d_4 = mixed7 --> max_pooling2d_4
      let conv2d_72 = mixed7 --> conv2d_71 --> conv2d_72
      let mixed8 = Concatenate([conv2d_72, conv2d_76, max_pooling2d_4], name: "mixed8")
      let conv2d_85 = mixed8 --> average_pooling2d_8 --> conv2d_85
      let conv2d_82 = mixed8 --> conv2d_81 --> conv2d_82
      let conv2d_78 = mixed8 --> conv2d_78
      let conv2d_79 = conv2d_78 --> conv2d_79
      let conv2d_77 = mixed8 --> conv2d_77
      let conv2d_84 = conv2d_82 --> conv2d_84
      let conv2d_83 = conv2d_82 --> conv2d_83
      let conv2d_80 = conv2d_78 --> conv2d_80
      let mixed9 = Concatenate([conv2d_77, conv2d_79, conv2d_80, conv2d_83, conv2d_84, conv2d_85], name: "mixed9")
      let conv2d_87 = mixed9 --> conv2d_87
      let conv2d_86 = mixed9 --> conv2d_86
      let conv2d_88 = conv2d_87 --> conv2d_88
      let conv2d_89 = conv2d_87 --> conv2d_89
      let conv2d_91 = mixed9 --> conv2d_90 --> conv2d_91
      let conv2d_94 = mixed9 --> average_pooling2d_9 --> conv2d_94
      let conv2d_93 = conv2d_91 --> conv2d_93
      let conv2d_92 = conv2d_91 --> conv2d_92
      let mixed10 = Concatenate([conv2d_86, conv2d_88, conv2d_89, conv2d_92, conv2d_93, conv2d_94], name: "mixed10")
      let predictions = mixed10 --> avg_pool --> predictions
      let output = predictions --> Softmax()
      model = Model(input: input, output: output)
    }
    // end of autogenerated forge net generation code
    var success = false
    let inflightBuffers = 1
    
    if debug {
      success = model.compile(device: device, inflightBuffers: inflightBuffers) {
        name, count, type in ParameterLoaderBundle(name: name,
                                                   count: count,
                                                   prefix: "inception_v3-",
                                                   suffix: type == .weights ? ".weights" : ".biases",
                                                   ext: "bin")
      }
      
    } else {
      success = model.compile(device: device, inflightBuffers: 1) {
        name, count, type in ParameterLoaderRandom(count: count)
      }
    }
    // end of autogenerated forge net generation code
    
    if success {
      print(model.summary())
    }
    
    if debug {
      let testInputImage = loadTexture(named: "final1_299.jpg")
      //let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
      let dumper = DocumentDirectoryDumper(filePrefix: "inception_v3")
      model.debugEvaluate(commandQueue: commandQueue, device: device, input: nil, inputTexure: testInputImage!, dumper: dumper)
      if !dumper.shapes.saveAsJson(fileName: "shapes.json") {
        print("Error dumping shapes")
      }
      
      let probabilitiesImage = model.outputImage(inflightIndex: 0)
      let probabilities = probabilitiesImage.toFloatArray()
      assert(probabilities.count == 1000)
      print("probabilities: \(probabilitiesImage.toFloatArrayChannelsInterleaved())")
      
      typealias Prediction = (label: String, probability: Float, index: Int)
      var result = NeuralNetworkResult<Prediction>()
      result.predictions = probabilities.top(k: 5).map { x -> Prediction in (imageNetLabels[x.0], x.1, x.0) }
      print("predictions:\(result.predictions)")
    } else {
      let test_input_data_3x2_3c_1i : [[[[Float]]]] = [[[[11,12,13],
                                                         [14,15,16]],
                                                        [[21,22,23],
                                                         [24,25,26]],
                                                        [[31,32,33],
                                                         [34,35,36]]]]
      
      let testInputImage = MPSImage(device: device, images: test_input_data_3x2_3c_1i)
      let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
    }
  }
  
  func testVGG16(debug: Bool) {
    print("\(self).\(#function)")
    
    var model : Model
    // begin of autogenerated forge net generation code
    
    let relu = MPSCNNNeuronReLU(device: device, a: 0)
    let input = Input()
    let swap_channels = Custom(TransposeChannelsKernel(device: device, featureChannels: 3, permute: [2,1,0]), name: "rgb2bgr")
    let subtract_mean = Custom(SubtractMeanColor(device:device, red: 123.68, green: 116.779, blue: 103.939, scale: 255.0), name: "subtract_mean")
    let input_2 = input --> Resize(width: 224, height: 224) -->  swap_channels -->  subtract_mean
    let block1_conv1 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "block1_conv1")
    let block1_conv2 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "block1_conv2")
    let block1_pool = MaxPooling(kernel: (2, 2), stride: (2, 2), name: "block1_pool")
    let block2_conv1 = Convolution(kernel: (3, 3), channels: 128, activation: relu, name: "block2_conv1")
    let block2_conv2 = Convolution(kernel: (3, 3), channels: 128, activation: relu, name: "block2_conv2")
    let block2_pool = MaxPooling(kernel: (2, 2), stride: (2, 2), name: "block2_pool")
    let block3_conv1 = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "block3_conv1")
    let block3_conv2 = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "block3_conv2")
    let block3_conv3 = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "block3_conv3")
    let block3_pool = MaxPooling(kernel: (2, 2), stride: (2, 2), name: "block3_pool")
    let block4_conv1 = Convolution(kernel: (3, 3), channels: 512, activation: relu, name: "block4_conv1")
    let block4_conv2 = Convolution(kernel: (3, 3), channels: 512, activation: relu, name: "block4_conv2")
    let block4_conv3 = Convolution(kernel: (3, 3), channels: 512, activation: relu, name: "block4_conv3")
    let block4_pool = MaxPooling(kernel: (2, 2), stride: (2, 2), name: "block4_pool")
    let block5_conv1 = Convolution(kernel: (3, 3), channels: 512, activation: relu, name: "block5_conv1")
    let block5_conv2 = Convolution(kernel: (3, 3), channels: 512, activation: relu, name: "block5_conv2")
    let block5_conv3 = Convolution(kernel: (3, 3), channels: 512, activation: relu, name: "block5_conv3")
    let block5_pool = MaxPooling(kernel: (2, 2), stride: (2, 2), name: "block5_pool")
    let fc1 = Dense(neurons: 4096, activation: relu, name: "fc1")
    let fc2 = Dense(neurons: 4096, activation: relu, name: "fc2")
    let predictions = Dense(neurons: 1000, name: "predictions")
    
    do {
      let predictions = input_2 --> block1_conv1 --> block1_conv2 --> block1_pool --> block2_conv1
        --> block2_conv2 --> block2_pool --> block3_conv1 --> block3_conv2 --> block3_conv3 --> block3_pool
        --> block4_conv1 --> block4_conv2 --> block4_conv3 --> block4_pool --> block5_conv1 --> block5_conv2 --> block5_conv3 --> block5_pool --> fc1 --> fc2 --> predictions
      let output = predictions --> Softmax()
      model = Model(input: input, output: output)
    }
    
    // end of autogenerated forge net generation code
    var success = false
    let inflightBuffers = 1
    
    if debug {
      success = model.compile(device: device, inflightBuffers: inflightBuffers) {
        
        name, count, type in ParameterLoaderBundle(name: name,
                                                   count: count,
                                                   prefix: "vgg_16-",
                                                   suffix: type == .weights ? ".weights" : ".biases",
                                                   ext: "bin")
      }
    } else {
      success = model.compile(device: device, inflightBuffers: 1) {
        name, count, type in ParameterLoaderRandom(count: count)
      }
    }
    // end of autogenerated forge net generation code
    
    if success {
      print(model.summary())
    }
    
    if debug {
      let testInputImage = loadTexture(named: "final1-224.jpg")
      //let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
      let dumper = DocumentDirectoryDumper(filePrefix: "vgg_16")
      model.debugEvaluate(commandQueue: commandQueue, device: device, input: nil, inputTexure: testInputImage!, dumper: dumper)

      if !dumper.shapes.saveAsJson(fileName: "shapes.json") {
        print("Error dumping shapes")
      }
      
      let probabilitiesImage = model.outputImage(inflightIndex: 0)
      let probabilities = probabilitiesImage.toFloatArray()
      assert(probabilities.count == 1000)
      print("probabilities: \(probabilitiesImage.toFloatArrayChannelsInterleaved())")
      
      typealias Prediction = (label: String, probability: Float, index: Int)
      var result = NeuralNetworkResult<Prediction>()
      result.predictions = probabilities.top(k: 5).map { x -> Prediction in (imageNetLabels[x.0], x.1, x.0) }
      print("predictions:\(result.predictions)")
    } else {
      let test_input_data_3x2_3c_1i : [[[[Float]]]] = [[[[11,12,13],
                                                         [14,15,16]],
                                                        [[21,22,23],
                                                         [24,25,26]],
                                                        [[31,32,33],
                                                         [34,35,36]]]]
      
      let testInputImage = MPSImage(device: device, images: test_input_data_3x2_3c_1i)
      let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
    }
  }

  
  let Inception3_labels = [
    "",
    "kit fox, Vulpes macrotis",
    "English setter",
    "Siberian husky",
    "Australian terrier",
    "English springer, English springer spaniel",
    "grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus",
    "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
    "Egyptian cat",
    "ibex, Capra ibex",
    "Persian cat",
    "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
    "gazelle",
    "porcupine, hedgehog",
    "sea lion",
    "malamute, malemute, Alaskan malamute",
    "badger",
    "Great Dane",
    "Walker hound, Walker foxhound",
    "Welsh springer spaniel",
    "whippet",
    "Scottish deerhound, deerhound",
    "killer whale, killer, orca, grampus, sea wolf, Orcinus orca",
    "mink",
    "African elephant, Loxodonta africana",
    "Weimaraner",
    "soft-coated wheaten terrier",
    "Dandie Dinmont, Dandie Dinmont terrier",
    "red wolf, maned wolf, Canis rufus, Canis niger",
    "Old English sheepdog, bobtail",
    "jaguar, panther, Panthera onca, Felis onca",
    "otterhound, otter hound",
    "bloodhound, sleuthhound",
    "Airedale, Airedale terrier",
    "hyena, hyaena",
    "meerkat, mierkat",
    "giant schnauzer",
    "titi, titi monkey",
    "three-toed sloth, ai, Bradypus tridactylus",
    "sorrel",
    "black-footed ferret, ferret, Mustela nigripes",
    "dalmatian, coach dog, carriage dog",
    "black-and-tan coonhound",
    "papillon",
    "skunk, polecat, wood pussy",
    "Staffordshire bullterrier, Staffordshire bull terrier",
    "Mexican hairless",
    "Bouvier des Flandres, Bouviers des Flandres",
    "weasel",
    "miniature poodle",
    "Cardigan, Cardigan Welsh corgi",
    "malinois",
    "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
    "fox squirrel, eastern fox squirrel, Sciurus niger",
    "colobus, colobus monkey",
    "tiger cat",
    "Lhasa, Lhasa apso",
    "impala, Aepyceros melampus",
    "coyote, prairie wolf, brush wolf, Canis latrans",
    "Yorkshire terrier",
    "Newfoundland, Newfoundland dog",
    "brown bear, bruin, Ursus arctos",
    "red fox, Vulpes vulpes",
    "Norwegian elkhound, elkhound",
    "Rottweiler",
    "hartebeest",
    "Saluki, gazelle hound",
    "grey fox, gray fox, Urocyon cinereoargenteus",
    "schipperke",
    "Pekinese, Pekingese, Peke",
    "Brabancon griffon",
    "West Highland white terrier",
    "Sealyham terrier, Sealyham",
    "guenon, guenon monkey",
    "mongoose",
    "indri, indris, Indri indri, Indri brevicaudatus",
    "tiger, Panthera tigris",
    "Irish wolfhound",
    "wild boar, boar, Sus scrofa",
    "EntleBucher",
    "zebra",
    "ram, tup",
    "French bulldog",
    "orangutan, orang, orangutang, Pongo pygmaeus",
    "basenji",
    "leopard, Panthera pardus",
    "Bernese mountain dog",
    "Maltese dog, Maltese terrier, Maltese",
    "Norfolk terrier",
    "toy terrier",
    "vizsla, Hungarian pointer",
    "cairn, cairn terrier",
    "squirrel monkey, Saimiri sciureus",
    "groenendael",
    "clumber, clumber spaniel",
    "Siamese cat, Siamese",
    "chimpanzee, chimp, Pan troglodytes",
    "komondor",
    "Afghan hound, Afghan",
    "Japanese spaniel",
    "proboscis monkey, Nasalis larvatus",
    "guinea pig, Cavia cobaya",
    "white wolf, Arctic wolf, Canis lupus tundrarum",
    "ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus",
    "gorilla, Gorilla gorilla",
    "borzoi, Russian wolfhound",
    "toy poodle",
    "Kerry blue terrier",
    "ox",
    "Scotch terrier, Scottish terrier, Scottie",
    "Tibetan mastiff",
    "spider monkey, Ateles geoffroyi",
    "Doberman, Doberman pinscher",
    "Boston bull, Boston terrier",
    "Greater Swiss Mountain dog",
    "Appenzeller",
    "Shih-Tzu",
    "Irish water spaniel",
    "Pomeranian",
    "Bedlington terrier",
    "warthog",
    "Arabian camel, dromedary, Camelus dromedarius",
    "siamang, Hylobates syndactylus, Symphalangus syndactylus",
    "miniature schnauzer",
    "collie",
    "golden retriever",
    "Irish terrier",
    "affenpinscher, monkey pinscher, monkey dog",
    "Border collie",
    "hare",
    "boxer",
    "silky terrier, Sydney silky",
    "beagle",
    "Leonberg",
    "German short-haired pointer",
    "patas, hussar monkey, Erythrocebus patas",
    "dhole, Cuon alpinus",
    "baboon",
    "macaque",
    "Chesapeake Bay retriever",
    "bull mastiff",
    "kuvasz",
    "capuchin, ringtail, Cebus capucinus",
    "pug, pug-dog",
    "curly-coated retriever",
    "Norwich terrier",
    "flat-coated retriever",
    "hog, pig, grunter, squealer, Sus scrofa",
    "keeshond",
    "Eskimo dog, husky",
    "Brittany spaniel",
    "standard poodle",
    "Lakeland terrier",
    "snow leopard, ounce, Panthera uncia",
    "Gordon setter",
    "dingo, warrigal, warragal, Canis dingo",
    "standard schnauzer",
    "hamster",
    "Tibetan terrier, chrysanthemum dog",
    "Arctic fox, white fox, Alopex lagopus",
    "wire-haired fox terrier",
    "basset, basset hound",
    "water buffalo, water ox, Asiatic buffalo, Bubalus bubalis",
    "American black bear, black bear, Ursus americanus, Euarctos americanus",
    "Angora, Angora rabbit",
    "bison",
    "howler monkey, howler",
    "hippopotamus, hippo, river horse, Hippopotamus amphibius",
    "chow, chow chow",
    "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca",
    "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier",
    "Shetland sheepdog, Shetland sheep dog, Shetland",
    "Great Pyrenees",
    "Chihuahua",
    "tabby, tabby cat",
    "marmoset",
    "Labrador retriever",
    "Saint Bernard, St Bernard",
    "armadillo",
    "Samoyed, Samoyede",
    "bluetick",
    "redbone",
    "polecat, fitch, foulmart, foumart, Mustela putorius",
    "marmot",
    "kelpie",
    "gibbon, Hylobates lar",
    "llama",
    "miniature pinscher",
    "wood rabbit, cottontail, cottontail rabbit",
    "Italian greyhound",
    "lion, king of beasts, Panthera leo",
    "cocker spaniel, English cocker spaniel, cocker",
    "Irish setter, red setter",
    "dugong, Dugong dugon",
    "Indian elephant, Elephas maximus",
    "beaver",
    "Sussex spaniel",
    "Pembroke, Pembroke Welsh corgi",
    "Blenheim spaniel",
    "Madagascar cat, ring-tailed lemur, Lemur catta",
    "Rhodesian ridgeback",
    "lynx, catamount",
    "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
    "langur",
    "Ibizan hound, Ibizan Podenco",
    "timber wolf, grey wolf, gray wolf, Canis lupus",
    "cheetah, chetah, Acinonyx jubatus",
    "English foxhound",
    "briard",
    "sloth bear, Melursus ursinus, Ursus ursinus",
    "Border terrier",
    "German shepherd, German shepherd dog, German police dog, alsatian",
    "otter",
    "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
    "tusker",
    "echidna, spiny anteater, anteater",
    "wallaby, brush kangaroo",
    "platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus",
    "wombat",
    "revolver, six-gun, six-shooter",
    "umbrella",
    "schooner",
    "soccer ball",
    "accordion, piano accordion, squeeze box",
    "ant, emmet, pismire",
    "starfish, sea star",
    "chambered nautilus, pearly nautilus, nautilus",
    "grand piano, grand",
    "laptop, laptop computer",
    "strawberry",
    "airliner",
    "warplane, military plane",
    "airship, dirigible",
    "balloon",
    "space shuttle",
    "fireboat",
    "gondola",
    "speedboat",
    "lifeboat",
    "canoe",
    "yawl",
    "catamaran",
    "trimaran",
    "container ship, containership, container vessel",
    "liner, ocean liner",
    "pirate, pirate ship",
    "aircraft carrier, carrier, flattop, attack aircraft carrier",
    "submarine, pigboat, sub, U-boat",
    "wreck",
    "half track",
    "tank, army tank, armored combat vehicle, armoured combat vehicle",
    "missile",
    "bobsled, bobsleigh, bob",
    "dogsled, dog sled, dog sleigh",
    "bicycle-built-for-two, tandem bicycle, tandem",
    "mountain bike, all-terrain bike, off-roader",
    "freight car",
    "passenger car, coach, carriage",
    "barrow, garden cart, lawn cart, wheelbarrow",
    "shopping cart",
    "motor scooter, scooter",
    "forklift",
    "electric locomotive",
    "steam locomotive",
    "amphibian, amphibious vehicle",
    "ambulance",
    "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
    "cab, hack, taxi, taxicab",
    "convertible",
    "jeep, landrover",
    "limousine, limo",
    "minivan",
    "Model T",
    "racer, race car, racing car",
    "sports car, sport car",
    "go-kart",
    "golfcart, golf cart",
    "moped",
    "snowplow, snowplough",
    "fire engine, fire truck",
    "garbage truck, dustcart",
    "pickup, pickup truck",
    "tow truck, tow car, wrecker",
    "trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi",
    "moving van",
    "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
    "recreational vehicle, RV, R.V.",
    "streetcar, tram, tramcar, trolley, trolley car",
    "snowmobile",
    "tractor",
    "mobile home, manufactured home",
    "tricycle, trike, velocipede",
    "unicycle, monocycle",
    "horse cart, horse-cart",
    "jinrikisha, ricksha, rickshaw",
    "oxcart",
    "bassinet",
    "cradle",
    "crib, cot",
    "four-poster",
    "bookcase",
    "china cabinet, china closet",
    "medicine chest, medicine cabinet",
    "chiffonier, commode",
    "table lamp",
    "file, file cabinet, filing cabinet",
    "park bench",
    "barber chair",
    "throne",
    "folding chair",
    "rocking chair, rocker",
    "studio couch, day bed",
    "toilet seat",
    "desk",
    "pool table, billiard table, snooker table",
    "dining table, board",
    "entertainment center",
    "wardrobe, closet, press",
    "Granny Smith",
    "orange",
    "lemon",
    "fig",
    "pineapple, ananas",
    "banana",
    "jackfruit, jak, jack",
    "custard apple",
    "pomegranate",
    "acorn",
    "hip, rose hip, rosehip",
    "ear, spike, capitulum",
    "rapeseed",
    "corn",
    "buckeye, horse chestnut, conker",
    "organ, pipe organ",
    "upright, upright piano",
    "chime, bell, gong",
    "drum, membranophone, tympan",
    "gong, tam-tam",
    "maraca",
    "marimba, xylophone",
    "steel drum",
    "banjo",
    "cello, violoncello",
    "violin, fiddle",
    "harp",
    "acoustic guitar",
    "electric guitar",
    "cornet, horn, trumpet, trump",
    "French horn, horn",
    "trombone",
    "harmonica, mouth organ, harp, mouth harp",
    "ocarina, sweet potato",
    "panpipe, pandean pipe, syrinx",
    "bassoon",
    "oboe, hautboy, hautbois",
    "sax, saxophone",
    "flute, transverse flute",
    "daisy",
    "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
    "cliff, drop, drop-off",
    "valley, vale",
    "alp",
    "volcano",
    "promontory, headland, head, foreland",
    "sandbar, sand bar",
    "coral reef",
    "lakeside, lakeshore",
    "seashore, coast, seacoast, sea-coast",
    "geyser",
    "hatchet",
    "cleaver, meat cleaver, chopper",
    "letter opener, paper knife, paperknife",
    "plane, carpenter's plane, woodworking plane",
    "power drill",
    "lawn mower, mower",
    "hammer",
    "corkscrew, bottle screw",
    "can opener, tin opener",
    "plunger, plumber's helper",
    "screwdriver",
    "shovel",
    "plow, plough",
    "chain saw, chainsaw",
    "cock",
    "hen",
    "ostrich, Struthio camelus",
    "brambling, Fringilla montifringilla",
    "goldfinch, Carduelis carduelis",
    "house finch, linnet, Carpodacus mexicanus",
    "junco, snowbird",
    "indigo bunting, indigo finch, indigo bird, Passerina cyanea",
    "robin, American robin, Turdus migratorius",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "water ouzel, dipper",
    "kite",
    "bald eagle, American eagle, Haliaeetus leucocephalus",
    "vulture",
    "great grey owl, great gray owl, Strix nebulosa",
    "black grouse",
    "ptarmigan",
    "ruffed grouse, partridge, Bonasa umbellus",
    "prairie chicken, prairie grouse, prairie fowl",
    "peacock",
    "quail",
    "partridge",
    "African grey, African gray, Psittacus erithacus",
    "macaw",
    "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
    "lorikeet",
    "coucal",
    "bee eater",
    "hornbill",
    "hummingbird",
    "jacamar",
    "toucan",
    "drake",
    "red-breasted merganser, Mergus serrator",
    "goose",
    "black swan, Cygnus atratus",
    "white stork, Ciconia ciconia",
    "black stork, Ciconia nigra",
    "spoonbill",
    "flamingo",
    "American egret, great white heron, Egretta albus",
    "little blue heron, Egretta caerulea",
    "bittern",
    "crane",
    "limpkin, Aramus pictus",
    "American coot, marsh hen, mud hen, water hen, Fulica americana",
    "bustard",
    "ruddy turnstone, Arenaria interpres",
    "red-backed sandpiper, dunlin, Erolia alpina",
    "redshank, Tringa totanus",
    "dowitcher",
    "oystercatcher, oyster catcher",
    "European gallinule, Porphyrio porphyrio",
    "pelican",
    "king penguin, Aptenodytes patagonica",
    "albatross, mollymawk",
    "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
    "tiger shark, Galeocerdo cuvieri",
    "hammerhead, hammerhead shark",
    "electric ray, crampfish, numbfish, torpedo",
    "stingray",
    "barracouta, snoek",
    "coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch",
    "tench, Tinca tinca",
    "goldfish, Carassius auratus",
    "eel",
    "rock beauty, Holocanthus tricolor",
    "anemone fish",
    "lionfish",
    "puffer, pufferfish, blowfish, globefish",
    "sturgeon",
    "gar, garfish, garpike, billfish, Lepisosteus osseus",
    "loggerhead, loggerhead turtle, Caretta caretta",
    "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
    "mud turtle",
    "terrapin",
    "box turtle, box tortoise",
    "banded gecko",
    "common iguana, iguana, Iguana iguana",
    "American chameleon, anole, Anolis carolinensis",
    "whiptail, whiptail lizard",
    "agama",
    "frilled lizard, Chlamydosaurus kingi",
    "alligator lizard",
    "Gila monster, Heloderma suspectum",
    "green lizard, Lacerta viridis",
    "African chameleon, Chamaeleo chamaeleon",
    "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
    "triceratops",
    "African crocodile, Nile crocodile, Crocodylus niloticus",
    "American alligator, Alligator mississipiensis",
    "thunder snake, worm snake, Carphophis amoenus",
    "ringneck snake, ring-necked snake, ring snake",
    "hognose snake, puff adder, sand viper",
    "green snake, grass snake",
    "king snake, kingsnake",
    "garter snake, grass snake",
    "water snake",
    "vine snake",
    "night snake, Hypsiglena torquata",
    "boa constrictor, Constrictor constrictor",
    "rock python, rock snake, Python sebae",
    "Indian cobra, Naja naja",
    "green mamba",
    "sea snake",
    "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
    "diamondback, diamondback rattlesnake, Crotalus adamanteus",
    "sidewinder, horned rattlesnake, Crotalus cerastes",
    "European fire salamander, Salamandra salamandra",
    "common newt, Triturus vulgaris",
    "eft",
    "spotted salamander, Ambystoma maculatum",
    "axolotl, mud puppy, Ambystoma mexicanum",
    "bullfrog, Rana catesbeiana",
    "tree frog, tree-frog",
    "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
    "whistle",
    "wing",
    "paintbrush",
    "hand blower, blow dryer, blow drier, hair dryer, hair drier",
    "oxygen mask",
    "snorkel",
    "loudspeaker, speaker, speaker unit, loudspeaker system, speaker system",
    "microphone, mike",
    "screen, CRT screen",
    "mouse, computer mouse",
    "electric fan, blower",
    "oil filter",
    "strainer",
    "space heater",
    "stove",
    "guillotine",
    "barometer",
    "rule, ruler",
    "odometer, hodometer, mileometer, milometer",
    "scale, weighing machine",
    "analog clock",
    "digital clock",
    "wall clock",
    "hourglass",
    "sundial",
    "parking meter",
    "stopwatch, stop watch",
    "digital watch",
    "stethoscope",
    "syringe",
    "magnetic compass",
    "binoculars, field glasses, opera glasses",
    "projector",
    "sunglasses, dark glasses, shades",
    "loupe, jeweler's loupe",
    "radio telescope, radio reflector",
    "bow",
    "cannon",
    "assault rifle, assault gun",
    "rifle",
    "projectile, missile",
    "computer keyboard, keypad",
    "typewriter keyboard",
    "crane",
    "lighter, light, igniter, ignitor",
    "abacus",
    "cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM",
    "slide rule, slipstick",
    "desktop computer",
    "hand-held computer, hand-held microcomputer",
    "notebook, notebook computer",
    "web site, website, internet site, site",
    "harvester, reaper",
    "thresher, thrasher, threshing machine",
    "printer",
    "slot, one-armed bandit",
    "vending machine",
    "sewing machine",
    "joystick",
    "switch, electric switch, electrical switch",
    "hook, claw",
    "car wheel",
    "paddlewheel, paddle wheel",
    "pinwheel",
    "potter's wheel",
    "gas pump, gasoline pump, petrol pump, island dispenser",
    "carousel, carrousel, merry-go-round, roundabout, whirligig",
    "swing",
    "reel",
    "radiator",
    "puck, hockey puck",
    "hard disc, hard disk, fixed disk",
    "sunglass",
    "pick, plectrum, plectron",
    "car mirror",
    "solar dish, solar collector, solar furnace",
    "remote control, remote",
    "disk brake, disc brake",
    "buckle",
    "hair slide",
    "knot",
    "combination lock",
    "padlock",
    "nail",
    "safety pin",
    "screw",
    "muzzle",
    "seat belt, seatbelt",
    "ski",
    "candle, taper, wax light",
    "jack-o'-lantern",
    "spotlight, spot",
    "torch",
    "neck brace",
    "pier",
    "tripod",
    "maypole",
    "mousetrap",
    "spider web, spider's web",
    "trilobite",
    "harvestman, daddy longlegs, Phalangium opilio",
    "scorpion",
    "black and gold garden spider, Argiope aurantia",
    "barn spider, Araneus cavaticus",
    "garden spider, Aranea diademata",
    "black widow, Latrodectus mactans",
    "tarantula",
    "wolf spider, hunting spider",
    "tick",
    "centipede",
    "isopod",
    "Dungeness crab, Cancer magister",
    "rock crab, Cancer irroratus",
    "fiddler crab",
    "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
    "American lobster, Northern lobster, Maine lobster, Homarus americanus",
    "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
    "crayfish, crawfish, crawdad, crawdaddy",
    "hermit crab",
    "tiger beetle",
    "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
    "ground beetle, carabid beetle",
    "long-horned beetle, longicorn, longicorn beetle",
    "leaf beetle, chrysomelid",
    "dung beetle",
    "rhinoceros beetle",
    "weevil",
    "fly",
    "bee",
    "grasshopper, hopper",
    "cricket",
    "walking stick, walkingstick, stick insect",
    "cockroach, roach",
    "mantis, mantid",
    "cicada, cicala",
    "leafhopper",
    "lacewing, lacewing fly",
    "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    "damselfly",
    "admiral",
    "ringlet, ringlet butterfly",
    "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
    "cabbage butterfly",
    "sulphur butterfly, sulfur butterfly",
    "lycaenid, lycaenid butterfly",
    "jellyfish",
    "sea anemone, anemone",
    "brain coral",
    "flatworm, platyhelminth",
    "nematode, nematode worm, roundworm",
    "conch",
    "snail",
    "slug",
    "sea slug, nudibranch",
    "chiton, coat-of-mail shell, sea cradle, polyplacophore",
    "sea urchin",
    "sea cucumber, holothurian",
    "iron, smoothing iron",
    "espresso maker",
    "microwave, microwave oven",
    "Dutch oven",
    "rotisserie",
    "toaster",
    "waffle iron",
    "vacuum, vacuum cleaner",
    "dishwasher, dish washer, dishwashing machine",
    "refrigerator, icebox",
    "washer, automatic washer, washing machine",
    "Crock Pot",
    "frying pan, frypan, skillet",
    "wok",
    "caldron, cauldron",
    "coffeepot",
    "teapot",
    "spatula",
    "altar",
    "triumphal arch",
    "patio, terrace",
    "steel arch bridge",
    "suspension bridge",
    "viaduct",
    "barn",
    "greenhouse, nursery, glasshouse",
    "palace",
    "monastery",
    "library",
    "apiary, bee house",
    "boathouse",
    "church, church building",
    "mosque",
    "stupa, tope",
    "planetarium",
    "restaurant, eating house, eating place, eatery",
    "cinema, movie theater, movie theatre, movie house, picture palace",
    "home theater, home theatre",
    "lumbermill, sawmill",
    "coil, spiral, volute, whorl, helix",
    "obelisk",
    "totem pole",
    "castle",
    "prison, prison house",
    "grocery store, grocery, food market, market",
    "bakery, bakeshop, bakehouse",
    "barbershop",
    "bookshop, bookstore, bookstall",
    "butcher shop, meat market",
    "confectionery, confectionary, candy store",
    "shoe shop, shoe-shop, shoe store",
    "tobacco shop, tobacconist shop, tobacconist",
    "toyshop",
    "fountain",
    "cliff dwelling",
    "yurt",
    "dock, dockage, docking facility",
    "brass, memorial tablet, plaque",
    "megalith, megalithic structure",
    "bannister, banister, balustrade, balusters, handrail",
    "breakwater, groin, groyne, mole, bulwark, seawall, jetty",
    "dam, dike, dyke",
    "chainlink fence",
    "picket fence, paling",
    "worm fence, snake fence, snake-rail fence, Virginia fence",
    "stone wall",
    "grille, radiator grille",
    "sliding door",
    "turnstile",
    "mountain tent",
    "scoreboard",
    "honeycomb",
    "plate rack",
    "pedestal, plinth, footstall",
    "beacon, lighthouse, beacon light, pharos",
    "mashed potato",
    "bell pepper",
    "head cabbage",
    "broccoli",
    "cauliflower",
    "zucchini, courgette",
    "spaghetti squash",
    "acorn squash",
    "butternut squash",
    "cucumber, cuke",
    "artichoke, globe artichoke",
    "cardoon",
    "mushroom",
    "shower curtain",
    "jean, blue jean, denim",
    "carton",
    "handkerchief, hankie, hanky, hankey",
    "sandal",
    "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
    "safe",
    "plate",
    "necklace",
    "croquet ball",
    "fur coat",
    "thimble",
    "pajama, pyjama, pj's, jammies",
    "running shoe",
    "cocktail shaker",
    "chest",
    "manhole cover",
    "modem",
    "tub, vat",
    "tray",
    "balance beam, beam",
    "bagel, beigel",
    "prayer rug, prayer mat",
    "kimono",
    "hot pot, hotpot",
    "whiskey jug",
    "knee pad",
    "book jacket, dust cover, dust jacket, dust wrapper",
    "spindle",
    "ski mask",
    "beer bottle",
    "crash helmet",
    "bottlecap",
    "tile roof",
    "mask",
    "maillot",
    "Petri dish",
    "football helmet",
    "bathing cap, swimming cap",
    "teddy, teddy bear",
    "holster",
    "pop bottle, soda bottle",
    "photocopier",
    "vestment",
    "crossword puzzle, crossword",
    "golf ball",
    "trifle",
    "suit, suit of clothes",
    "water tower",
    "feather boa, boa",
    "cloak",
    "red wine",
    "drumstick",
    "shield, buckler",
    "Christmas stocking",
    "hoopskirt, crinoline",
    "menu",
    "stage",
    "bonnet, poke bonnet",
    "meat loaf, meatloaf",
    "baseball",
    "face powder",
    "scabbard",
    "sunscreen, sunblock, sun blocker",
    "beer glass",
    "hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa",
    "guacamole",
    "lampshade, lamp shade",
    "wool, woolen, woollen",
    "hay",
    "bow tie, bow-tie, bowtie",
    "mailbag, postbag",
    "water jug",
    "bucket, pail",
    "dishrag, dishcloth",
    "soup bowl",
    "eggnog",
    "mortar",
    "trench coat",
    "paddle, boat paddle",
    "chain",
    "swab, swob, mop",
    "mixing bowl",
    "potpie",
    "wine bottle",
    "shoji",
    "bulletproof vest",
    "drilling platform, offshore rig",
    "binder, ring-binder",
    "cardigan",
    "sweatshirt",
    "pot, flowerpot",
    "birdhouse",
    "hamper",
    "ping-pong ball",
    "pencil box, pencil case",
    "pay-phone, pay-station",
    "consomme",
    "apron",
    "punching bag, punch bag, punching ball, punchball",
    "backpack, back pack, knapsack, packsack, rucksack, haversack",
    "groom, bridegroom",
    "bearskin, busby, shako",
    "pencil sharpener",
    "broom",
    "mosquito net",
    "abaya",
    "mortarboard",
    "poncho",
    "crutch",
    "Polaroid camera, Polaroid Land camera",
    "space bar",
    "cup",
    "racket, racquet",
    "traffic light, traffic signal, stoplight",
    "quill, quill pen",
    "radio, wireless",
    "dough",
    "cuirass",
    "military uniform",
    "lipstick, lip rouge",
    "shower cap",
    "monitor",
    "oscilloscope, scope, cathode-ray oscilloscope, CRO",
    "mitten",
    "brassiere, bra, bandeau",
    "French loaf",
    "vase",
    "milk can",
    "rugby ball",
    "paper towel",
    "earthstar",
    "envelope",
    "miniskirt, mini",
    "cowboy hat, ten-gallon hat",
    "trolleybus, trolley coach, trackless trolley",
    "perfume, essence",
    "bathtub, bathing tub, bath, tub",
    "hotdog, hot dog, red hot",
    "coral fungus",
    "bullet train, bullet",
    "pillow",
    "toilet tissue, toilet paper, bathroom tissue",
    "cassette",
    "carpenter's kit, tool kit",
    "ladle",
    "stinkhorn, carrion fungus",
    "lotion",
    "hair spray",
    "academic gown, academic robe, judge's robe",
    "dome",
    "crate",
    "wig",
    "burrito",
    "pill bottle",
    "chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour",
    "theater curtain, theatre curtain",
    "window shade",
    "barrel, cask",
    "washbasin, handbasin, washbowl, lavabo, wash-hand basin",
    "ballpoint, ballpoint pen, ballpen, Biro",
    "basketball",
    "bath towel",
    "cowboy boot",
    "gown",
    "window screen",
    "agaric",
    "cellular telephone, cellular phone, cellphone, cell, mobile phone",
    "nipple",
    "barbell",
    "mailbox, letter box",
    "lab coat, laboratory coat",
    "fire screen, fireguard",
    "minibus",
    "packet",
    "maze, labyrinth",
    "pole",
    "horizontal bar, high bar",
    "sombrero",
    "pickelhaube",
    "rain barrel",
    "wallet, billfold, notecase, pocketbook",
    "cassette player",
    "comic book",
    "piggy bank, penny bank",
    "street sign",
    "bell cote, bell cot",
    "fountain pen",
    "Windsor tie",
    "volleyball",
    "overskirt",
    "sarong",
    "purse",
    "bolo tie, bolo, bola tie, bola",
    "bib",
    "parachute, chute",
    "sleeping bag",
    "television, television system",
    "swimming trunks, bathing trunks",
    "measuring cup",
    "espresso",
    "pizza, pizza pie",
    "breastplate, aegis, egis",
    "shopping basket",
    "wooden spoon",
    "saltshaker, salt shaker",
    "chocolate sauce, chocolate syrup",
    "ballplayer, baseball player",
    "goblet",
    "gyromitra",
    "stretcher",
    "water bottle",
    "dial telephone, dial phone",
    "soap dispenser",
    "jersey, T-shirt, tee shirt",
    "school bus",
    "jigsaw puzzle",
    "plastic bag",
    "reflex camera",
    "diaper, nappy, napkin",
    "Band Aid",
    "ice lolly, lolly, lollipop, popsicle",
    "velvet",
    "tennis ball",
    "gasmask, respirator, gas helmet",
    "doormat, welcome mat",
    "Loafer",
    "ice cream, icecream",
    "pretzel",
    "quilt, comforter, comfort, puff",
    "maillot, tank suit",
    "tape player",
    "clog, geta, patten, sabot",
    "iPod",
    "bolete",
    "scuba diver",
    "pitcher, ewer",
    "matchstick",
    "bikini, two-piece",
    "sock",
    "CD player",
    "lens cap, lens cover",
    "thatch, thatched roof",
    "vault",
    "beaker",
    "bubble",
    "cheeseburger",
    "parallel bars, bars",
    "flagpole, flagstaff",
    "coffee mug",
    "rubber eraser, rubber, pencil eraser",
    "stole",
    "carbonara",
    "dumbbell"]
}
