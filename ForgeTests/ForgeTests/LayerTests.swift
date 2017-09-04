//
//  LayerTests.swift
//  ForgeTests
//
//  Created by Pavel Mayer on 02.09.17.
//  Copyright © 2017 Tognos GmbH. All rights reserved.
//

import Foundation
import Forge
import MetalPerformanceShaders

class LayerTests {
  
  func printLocation(_ location: String) {
    print(location)
  }
  
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
    let resultImage = model.evaluate(commandQueue: commandQueue, device: device, model: model, input: testInputImage)

    //print("resultImage=",resultImage)
    let resultData = resultImage.toFloatArray4D()
    //print("resultData=",resultData)

    let expectedOutputData : [[[[Float]]]] = [[[[4,7,10],
                                                [13,16,19]]]]
    if !areAlmostEqual(resultData, expectedOutputData, maxError: Float(1e-2), reportUnequal: printLocation) {
      fatalError("Assertion failed: \(resultData) not equal to \(expectedOutputData)")
    }
  }
  
  func testZeroPaddingLayer() {
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
    let resultImage = model.evaluate(commandQueue: commandQueue, device: device, model: model, input: testInputImage)
    
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
    let resultImage = model.evaluate(commandQueue: commandQueue, device: device, model: model, input: testInputImage)
    
    print("resultImage=",resultImage)
    let resultData = resultImage.toFloatArray4D()
    print("resultData=",resultData)
    
    let expectedOutputData : [[[[Float]]]] = [[[[1.0, 2.0],
                                                [5.0, 6.0]],
                                               [[21.0, 22.0],
                                                [25.0, 26.0]],
                                               [[31.0, 32.0],
                                                [35.0, 36.0]],
                                               [[41.0, 42.0],
                                                [45.0, 46.0]],
                                               [[3.0, 4.0],
                                                [7.0, 8.0]],
                                               [[23.0, 24.0],
                                                [27.0, 28.0]],
                                               [[33.0, 34.0],
                                                [37.0, 38.0]],
                                               [[43.0, 44.0],
                                                [47.0, 48.0]],
                                               [[9.0, 10.0 ],
                                                [13.0, 14.0]],
                                               [[29.0, 210.0],
                                                [213.0, 214.0]],
                                               [[39.0, 310.0],
                                                [313.0, 314.0]],
                                               [[49.0, 410.0],
                                                [413.0, 414.0]],
                                               [[11.0, 12.0],
                                                [15.0, 16.0]],
                                               [[211.0, 212.0],
                                                [215.0, 216.0]],
                                               [[311.0, 312.0],
                                                [315.0, 316.0]],
                                               [[411.0, 412.0],
                                                [415.0, 416.0]]]]
    if !areAlmostEqual(resultData, expectedOutputData, maxError: Float(1e-2), reportUnequal: printLocation) {
      fatalError("Assertion failed: \(resultData) not equal to \(expectedOutputData)")
    }
  }
  func testSimpleMerge() {
    print("\(self).\(#function)")
    
    //f(x) = a * x + b
    let linear1 = MPSCNNNeuronLinear(device: device, a: 2.0, b: 0.0)
    let linear2 = MPSCNNNeuronLinear(device: device, a: 3.0, b: 0.0)
    let input = Input(width: 3, height: 2, channels: 4)
    let activation1 = Activation(linear1, name: "activation1")
    let activation2 = Activation(linear2, name: "activation2")

    do {
      let activation1 = input --> activation1
      let activation2 = input --> activation2
      let merged = Merge([activation1, activation2])
      let output = merged --> Add(name:"adder")
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
      let resultImage = model.evaluate(commandQueue: commandQueue, device: device, model: model, input: testInputImage)
      
      //print("resultImage=",resultImage)
      let resultData = resultImage.toFloatArray4D()
      //print("resultData=",resultData)
      
      let expectedOutputData : [[[[Float]]]] = [[[[22, 24, 26],
                                                  [28, 30, 32]],
                                                 [[42, 44, 46],
                                                  [48, 50, 52]],
                                                 [[62, 64, 66],
                                                  [68, 70, 72]],
                                                 [[82, 84, 86],
                                                  [88, 90, 92]]]]
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
//      print("test_input_data_3x2_3c_1i:",test_input_data_3x2_3c_1i)
      let test_input_image_3x2_3c_1i = MPSImage(device: device, images: test_input_data_3x2_3c_1i)
      
      let test_read_data_3x2_3c_1i = test_input_image_3x2_3c_1i.toFloatArray4D()
//      print("test_read_data_3x2_3c_1i:",test_read_data_3x2_3c_1i)
      
      if !(areAlmostEqual(test_input_data_3x2_3c_1i, test_read_data_3x2_3c_1i, maxError: maxError, reportUnequal: printLocation)) {
        fatalError("Assertion failed: \(test_input_data_3x2_3c_1i) not equal to \(test_read_data_3x2_3c_1i)")
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
}
