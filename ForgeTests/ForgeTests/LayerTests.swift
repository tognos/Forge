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

func URLinDocumentDirectory(fileName: String) -> URL {
  let documentsDirectoryURL = try! FileManager().url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
  return documentsDirectoryURL.appendingPathComponent(fileName)
}

extension UIImage {
  func saveInDocumentDirectory(fileName: String) {
    print("Saving", fileName)
    do {
      try UIImageJPEGRepresentation(self,1.0)!.write(to: URLinDocumentDirectory(fileName: fileName))
      print("Image added successfully")
    } catch {
      print(error)
    }
  }
}

extension Data {
  func saveInDocumentDirectory(fileName: String) -> Bool {
    print("Saving data as ", fileName)
    do {
      try self.write(to: URLinDocumentDirectory(fileName: fileName))
      print("Data saved successfully")
      return true
    } catch {
      print(error)
      return false
    }
  }
}

// Swift can really need some polishing
protocol isFloat {}
extension Float : isFloat {}
extension Array where Element : isFloat {
  func saveInDocumentDirectory(fileName: String) -> Bool {
    return self.withUnsafeBytes({ (bufferptr) in
      let data = Data(bufferptr)
      return data.saveInDocumentDirectory(fileName: fileName)
    })
  }
}

public class DocumentDirectoryDumper : TensorDumper {
  let filePrefix : String
  public init(filePrefix: String) {
    self.filePrefix = filePrefix
  }
  public func dump(tensor: Tensor) -> Bool {
    let fileBase = filePrefix + "-" + tensor.shortId
    let rawData = tensor.image!.toFloatArrayChannelsFirst()
    let ok = rawData.saveInDocumentDirectory(fileName: fileBase+".floats")
    return ok
  }
}

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
  func testResNet(debug: Bool) {
    print("\(self).\(#function)")
    
    var model : Model
    // begin of autogenerated forge net generation code
    
    let relu = MPSCNNNeuronReLU(device: device, a: 0)
    let input = Input()
    let input_2 = input --> Resize(width: 224, height: 224)
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
    let fc1000 = Dense(neurons: 1000, name: "fc1000")
    
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
      let res3a_branch2c = activation_10 --> res3a_branch2a --> res3a_branch2b --> res3a_branch2c
      let res3a_branch1 = activation_10 --> res3a_branch1
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
    let success = model.compile(device: device, inflightBuffers: 1) {
      name, count, type in ParameterLoaderRandom(count: count)
    }
    
    // end of autogenerated forge net generation code
    
    if success {
      print(model.summary())
    }
    let test_input_data_3x2_3c_1i : [[[[Float]]]] = [[[[11,12,13],
                                                       [14,15,16]],
                                                      [[21,22,23],
                                                       [24,25,26]],
                                                      [[31,32,33],
                                                       [34,35,36]]]]
    
    let testInputImage = MPSImage(device: device, images: test_input_data_3x2_3c_1i)
    if debug {
      //let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
      model.debugEvaluate(commandQueue: commandQueue, device: device, input: testInputImage, dumper: DocumentDirectoryDumper(filePrefix: "resnet-50"))
    } else {
      let resultImage = model.evaluate(commandQueue: commandQueue, device: device, input: testInputImage)
    }
  }
}
