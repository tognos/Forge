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
    print("Saving data in document directory as '\(fileName)'")
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

func asJsonData(_ dictonary : Dictionary<String, Any>) -> Data? {
  do {
    let jsonData = try JSONSerialization.data(withJSONObject: dictonary, options: .prettyPrinted)
    //print(NSString(data: jsonData, encoding: 1)!)
    return jsonData
  } catch let error {
    print("Error serializing:",error)
  }
  return nil
}

extension Dictionary where Key == String, Value: Any {
  func asJson() -> Data? {
    return asJsonData(self)
  }
  func saveAsJson(fileName: String) -> Bool {
    let data = asJsonData(self)
    if data != nil {
      return data!.saveInDocumentDirectory(fileName: fileName)
    }
    return false
  }
}

public class DocumentDirectoryDumper : TensorDumper {
  let filePrefix : String
  public var shapes : [String: Any] = [:]
  public init(filePrefix: String) {
    self.filePrefix = filePrefix
  }
  public func dump(tensor: Tensor) -> Bool {
    var ok = false
    if tensor.shape.numImages == 1 {
      let fileName = filePrefix + "-" + tensor.shortId + ".floats"
      let rawData = tensor.image!.toFloatArrayChannelsInterleaved()
      ok = rawData.saveInDocumentDirectory(fileName: fileName)
      if ok {
        shapes[fileName] = tensor.shape.asArray
      }
      if tensor.typeName == "Concat" {
        // Also dump output images of input
        // shapeChannelsInterleaved = (numberOfImages, height, width, featureChannels)
        let iShape = tensor.image!.shapeChannelsInterleaved
        let reshaped = rawData.reshaped(iShape)
        print("DocumentDirectoryDumper: Tensor \(tensor.shortId) is Concat Tensor, saving partial data as output of inputs")
        for input in tensor.previous {
          
          let iSlice = sliceArray(reshaped, from: (0, 0, 0, input.destinationChannelOffset),
                                  size: (iShape.0, iShape.1, iShape.2, input.shape.channels))
          let fileName = filePrefix + "-" + input.shortId + ".floats"
          let rawSlice = flattened(iSlice)
          ok = rawSlice.saveInDocumentDirectory(fileName: fileName)

          if ok {
            shapes[fileName] = input.shape.asArray
          }
          print("DocumentDirectoryDumper: Saving output for Tensor \(input.shortId)")
        }
      }
    } else {
      // dump multiple images under the name of their outputs
      print("DocumentDirectoryDumper: Tensor \(tensor.shortId) has \(tensor.shape.numImages) images, saving as outputs of inputs")
      for input in tensor.previous {
        print("DocumentDirectoryDumper: Saving output for Tensor \(input.shortId)")
        let rawData = tensor.image!.toFloatArrayChannelsInterleaved(fromImage: input.destinationImageNumber, numImages:1)
        let fileName = filePrefix + "-" + input.shortId + ".floats"
        ok = rawData.saveInDocumentDirectory(fileName: fileName)
        if ok {
          shapes[fileName] = input.shape.asArray
        }
      }
    }
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
      let expected = [[[[101.0, 201.0, 301.0], [104.0, 204.0, 304.0]]], [[[102.0, 202.0, 302.0], [105.0, 205.0, 305.0]]], [[[103.0, 203.0, 303.0], [106.0, 206.0, 306.0]]]]
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
      result.predictions = probabilities.top(k: 5).map { x -> Prediction in (self.imageNetLabels[x.0], x.1, x.0) }
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
      result.predictions = probabilities.top(k: 5).map { x -> Prediction in (self.imageNetLabels[x.0], x.1, x.0) }
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
      result.predictions = probabilities.top(k: 5).map { x -> Prediction in (self.imageNetLabels[x.0], x.1, x.0) }
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

  
  let imageNetLabels = [
    "n01440764 tench, Tinca tinca",
    "n01443537 goldfish, Carassius auratus",
    "n01484850 great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
    "n01491361 tiger shark, Galeocerdo cuvieri",
    "n01494475 hammerhead, hammerhead shark",
    "n01496331 electric ray, crampfish, numbfish, torpedo",
    "n01498041 stingray",
    "n01514668 cock",
    "n01514859 hen",
    "n01518878 ostrich, Struthio camelus",
    "n01530575 brambling, Fringilla montifringilla",
    "n01531178 goldfinch, Carduelis carduelis",
    "n01532829 house finch, linnet, Carpodacus mexicanus",
    "n01534433 junco, snowbird",
    "n01537544 indigo bunting, indigo finch, indigo bird, Passerina cyanea",
    "n01558993 robin, American robin, Turdus migratorius",
    "n01560419 bulbul",
    "n01580077 jay",
    "n01582220 magpie",
    "n01592084 chickadee",
    "n01601694 water ouzel, dipper",
    "n01608432 kite",
    "n01614925 bald eagle, American eagle, Haliaeetus leucocephalus",
    "n01616318 vulture",
    "n01622779 great grey owl, great gray owl, Strix nebulosa",
    "n01629819 European fire salamander, Salamandra salamandra",
    "n01630670 common newt, Triturus vulgaris",
    "n01631663 eft",
    "n01632458 spotted salamander, Ambystoma maculatum",
    "n01632777 axolotl, mud puppy, Ambystoma mexicanum",
    "n01641577 bullfrog, Rana catesbeiana",
    "n01644373 tree frog, tree-frog",
    "n01644900 tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
    "n01664065 loggerhead, loggerhead turtle, Caretta caretta",
    "n01665541 leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
    "n01667114 mud turtle",
    "n01667778 terrapin",
    "n01669191 box turtle, box tortoise",
    "n01675722 banded gecko",
    "n01677366 common iguana, iguana, Iguana iguana",
    "n01682714 American chameleon, anole, Anolis carolinensis",
    "n01685808 whiptail, whiptail lizard",
    "n01687978 agama",
    "n01688243 frilled lizard, Chlamydosaurus kingi",
    "n01689811 alligator lizard",
    "n01692333 Gila monster, Heloderma suspectum",
    "n01693334 green lizard, Lacerta viridis",
    "n01694178 African chameleon, Chamaeleo chamaeleon",
    "n01695060 Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
    "n01697457 African crocodile, Nile crocodile, Crocodylus niloticus",
    "n01698640 American alligator, Alligator mississipiensis",
    "n01704323 triceratops",
    "n01728572 thunder snake, worm snake, Carphophis amoenus",
    "n01728920 ringneck snake, ring-necked snake, ring snake",
    "n01729322 hognose snake, puff adder, sand viper",
    "n01729977 green snake, grass snake",
    "n01734418 king snake, kingsnake",
    "n01735189 garter snake, grass snake",
    "n01737021 water snake",
    "n01739381 vine snake",
    "n01740131 night snake, Hypsiglena torquata",
    "n01742172 boa constrictor, Constrictor constrictor",
    "n01744401 rock python, rock snake, Python sebae",
    "n01748264 Indian cobra, Naja naja",
    "n01749939 green mamba",
    "n01751748 sea snake",
    "n01753488 horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
    "n01755581 diamondback, diamondback rattlesnake, Crotalus adamanteus",
    "n01756291 sidewinder, horned rattlesnake, Crotalus cerastes",
    "n01768244 trilobite",
    "n01770081 harvestman, daddy longlegs, Phalangium opilio",
    "n01770393 scorpion",
    "n01773157 black and gold garden spider, Argiope aurantia",
    "n01773549 barn spider, Araneus cavaticus",
    "n01773797 garden spider, Aranea diademata",
    "n01774384 black widow, Latrodectus mactans",
    "n01774750 tarantula",
    "n01775062 wolf spider, hunting spider",
    "n01776313 tick",
    "n01784675 centipede",
    "n01795545 black grouse",
    "n01796340 ptarmigan",
    "n01797886 ruffed grouse, partridge, Bonasa umbellus",
    "n01798484 prairie chicken, prairie grouse, prairie fowl",
    "n01806143 peacock",
    "n01806567 quail",
    "n01807496 partridge",
    "n01817953 African grey, African gray, Psittacus erithacus",
    "n01818515 macaw",
    "n01819313 sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
    "n01820546 lorikeet",
    "n01824575 coucal",
    "n01828970 bee eater",
    "n01829413 hornbill",
    "n01833805 hummingbird",
    "n01843065 jacamar",
    "n01843383 toucan",
    "n01847000 drake",
    "n01855032 red-breasted merganser, Mergus serrator",
    "n01855672 goose",
    "n01860187 black swan, Cygnus atratus",
    "n01871265 tusker",
    "n01872401 echidna, spiny anteater, anteater",
    "n01873310 platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus",
    "n01877812 wallaby, brush kangaroo",
    "n01882714 koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
    "n01883070 wombat",
    "n01910747 jellyfish",
    "n01914609 sea anemone, anemone",
    "n01917289 brain coral",
    "n01924916 flatworm, platyhelminth",
    "n01930112 nematode, nematode worm, roundworm",
    "n01943899 conch",
    "n01944390 snail",
    "n01945685 slug",
    "n01950731 sea slug, nudibranch",
    "n01955084 chiton, coat-of-mail shell, sea cradle, polyplacophore",
    "n01968897 chambered nautilus, pearly nautilus, nautilus",
    "n01978287 Dungeness crab, Cancer magister",
    "n01978455 rock crab, Cancer irroratus",
    "n01980166 fiddler crab",
    "n01981276 king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
    "n01983481 American lobster, Northern lobster, Maine lobster, Homarus americanus",
    "n01984695 spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
    "n01985128 crayfish, crawfish, crawdad, crawdaddy",
    "n01986214 hermit crab",
    "n01990800 isopod",
    "n02002556 white stork, Ciconia ciconia",
    "n02002724 black stork, Ciconia nigra",
    "n02006656 spoonbill",
    "n02007558 flamingo",
    "n02009229 little blue heron, Egretta caerulea",
    "n02009912 American egret, great white heron, Egretta albus",
    "n02011460 bittern",
    "n02012849 crane",
    "n02013706 limpkin, Aramus pictus",
    "n02017213 European gallinule, Porphyrio porphyrio",
    "n02018207 American coot, marsh hen, mud hen, water hen, Fulica americana",
    "n02018795 bustard",
    "n02025239 ruddy turnstone, Arenaria interpres",
    "n02027492 red-backed sandpiper, dunlin, Erolia alpina",
    "n02028035 redshank, Tringa totanus",
    "n02033041 dowitcher",
    "n02037110 oystercatcher, oyster catcher",
    "n02051845 pelican",
    "n02056570 king penguin, Aptenodytes patagonica",
    "n02058221 albatross, mollymawk",
    "n02066245 grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus",
    "n02071294 killer whale, killer, orca, grampus, sea wolf, Orcinus orca",
    "n02074367 dugong, Dugong dugon",
    "n02077923 sea lion",
    "n02085620 Chihuahua",
    "n02085782 Japanese spaniel",
    "n02085936 Maltese dog, Maltese terrier, Maltese",
    "n02086079 Pekinese, Pekingese, Peke",
    "n02086240 Shih-Tzu",
    "n02086646 Blenheim spaniel",
    "n02086910 papillon",
    "n02087046 toy terrier",
    "n02087394 Rhodesian ridgeback",
    "n02088094 Afghan hound, Afghan",
    "n02088238 basset, basset hound",
    "n02088364 beagle",
    "n02088466 bloodhound, sleuthhound",
    "n02088632 bluetick",
    "n02089078 black-and-tan coonhound",
    "n02089867 Walker hound, Walker foxhound",
    "n02089973 English foxhound",
    "n02090379 redbone",
    "n02090622 borzoi, Russian wolfhound",
    "n02090721 Irish wolfhound",
    "n02091032 Italian greyhound",
    "n02091134 whippet",
    "n02091244 Ibizan hound, Ibizan Podenco",
    "n02091467 Norwegian elkhound, elkhound",
    "n02091635 otterhound, otter hound",
    "n02091831 Saluki, gazelle hound",
    "n02092002 Scottish deerhound, deerhound",
    "n02092339 Weimaraner",
    "n02093256 Staffordshire bullterrier, Staffordshire bull terrier",
    "n02093428 American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier",
    "n02093647 Bedlington terrier",
    "n02093754 Border terrier",
    "n02093859 Kerry blue terrier",
    "n02093991 Irish terrier",
    "n02094114 Norfolk terrier",
    "n02094258 Norwich terrier",
    "n02094433 Yorkshire terrier",
    "n02095314 wire-haired fox terrier",
    "n02095570 Lakeland terrier",
    "n02095889 Sealyham terrier, Sealyham",
    "n02096051 Airedale, Airedale terrier",
    "n02096177 cairn, cairn terrier",
    "n02096294 Australian terrier",
    "n02096437 Dandie Dinmont, Dandie Dinmont terrier",
    "n02096585 Boston bull, Boston terrier",
    "n02097047 miniature schnauzer",
    "n02097130 giant schnauzer",
    "n02097209 standard schnauzer",
    "n02097298 Scotch terrier, Scottish terrier, Scottie",
    "n02097474 Tibetan terrier, chrysanthemum dog",
    "n02097658 silky terrier, Sydney silky",
    "n02098105 soft-coated wheaten terrier",
    "n02098286 West Highland white terrier",
    "n02098413 Lhasa, Lhasa apso",
    "n02099267 flat-coated retriever",
    "n02099429 curly-coated retriever",
    "n02099601 golden retriever",
    "n02099712 Labrador retriever",
    "n02099849 Chesapeake Bay retriever",
    "n02100236 German short-haired pointer",
    "n02100583 vizsla, Hungarian pointer",
    "n02100735 English setter",
    "n02100877 Irish setter, red setter",
    "n02101006 Gordon setter",
    "n02101388 Brittany spaniel",
    "n02101556 clumber, clumber spaniel",
    "n02102040 English springer, English springer spaniel",
    "n02102177 Welsh springer spaniel",
    "n02102318 cocker spaniel, English cocker spaniel, cocker",
    "n02102480 Sussex spaniel",
    "n02102973 Irish water spaniel",
    "n02104029 kuvasz",
    "n02104365 schipperke",
    "n02105056 groenendael",
    "n02105162 malinois",
    "n02105251 briard",
    "n02105412 kelpie",
    "n02105505 komondor",
    "n02105641 Old English sheepdog, bobtail",
    "n02105855 Shetland sheepdog, Shetland sheep dog, Shetland",
    "n02106030 collie",
    "n02106166 Border collie",
    "n02106382 Bouvier des Flandres, Bouviers des Flandres",
    "n02106550 Rottweiler",
    "n02106662 German shepherd, German shepherd dog, German police dog, alsatian",
    "n02107142 Doberman, Doberman pinscher",
    "n02107312 miniature pinscher",
    "n02107574 Greater Swiss Mountain dog",
    "n02107683 Bernese mountain dog",
    "n02107908 Appenzeller",
    "n02108000 EntleBucher",
    "n02108089 boxer",
    "n02108422 bull mastiff",
    "n02108551 Tibetan mastiff",
    "n02108915 French bulldog",
    "n02109047 Great Dane",
    "n02109525 Saint Bernard, St Bernard",
    "n02109961 Eskimo dog, husky",
    "n02110063 malamute, malemute, Alaskan malamute",
    "n02110185 Siberian husky",
    "n02110341 dalmatian, coach dog, carriage dog",
    "n02110627 affenpinscher, monkey pinscher, monkey dog",
    "n02110806 basenji",
    "n02110958 pug, pug-dog",
    "n02111129 Leonberg",
    "n02111277 Newfoundland, Newfoundland dog",
    "n02111500 Great Pyrenees",
    "n02111889 Samoyed, Samoyede",
    "n02112018 Pomeranian",
    "n02112137 chow, chow chow",
    "n02112350 keeshond",
    "n02112706 Brabancon griffon",
    "n02113023 Pembroke, Pembroke Welsh corgi",
    "n02113186 Cardigan, Cardigan Welsh corgi",
    "n02113624 toy poodle",
    "n02113712 miniature poodle",
    "n02113799 standard poodle",
    "n02113978 Mexican hairless",
    "n02114367 timber wolf, grey wolf, gray wolf, Canis lupus",
    "n02114548 white wolf, Arctic wolf, Canis lupus tundrarum",
    "n02114712 red wolf, maned wolf, Canis rufus, Canis niger",
    "n02114855 coyote, prairie wolf, brush wolf, Canis latrans",
    "n02115641 dingo, warrigal, warragal, Canis dingo",
    "n02115913 dhole, Cuon alpinus",
    "n02116738 African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
    "n02117135 hyena, hyaena",
    "n02119022 red fox, Vulpes vulpes",
    "n02119789 kit fox, Vulpes macrotis",
    "n02120079 Arctic fox, white fox, Alopex lagopus",
    "n02120505 grey fox, gray fox, Urocyon cinereoargenteus",
    "n02123045 tabby, tabby cat",
    "n02123159 tiger cat",
    "n02123394 Persian cat",
    "n02123597 Siamese cat, Siamese",
    "n02124075 Egyptian cat",
    "n02125311 cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
    "n02127052 lynx, catamount",
    "n02128385 leopard, Panthera pardus",
    "n02128757 snow leopard, ounce, Panthera uncia",
    "n02128925 jaguar, panther, Panthera onca, Felis onca",
    "n02129165 lion, king of beasts, Panthera leo",
    "n02129604 tiger, Panthera tigris",
    "n02130308 cheetah, chetah, Acinonyx jubatus",
    "n02132136 brown bear, bruin, Ursus arctos",
    "n02133161 American black bear, black bear, Ursus americanus, Euarctos americanus",
    "n02134084 ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus",
    "n02134418 sloth bear, Melursus ursinus, Ursus ursinus",
    "n02137549 mongoose",
    "n02138441 meerkat, mierkat",
    "n02165105 tiger beetle",
    "n02165456 ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
    "n02167151 ground beetle, carabid beetle",
    "n02168699 long-horned beetle, longicorn, longicorn beetle",
    "n02169497 leaf beetle, chrysomelid",
    "n02172182 dung beetle",
    "n02174001 rhinoceros beetle",
    "n02177972 weevil",
    "n02190166 fly",
    "n02206856 bee",
    "n02219486 ant, emmet, pismire",
    "n02226429 grasshopper, hopper",
    "n02229544 cricket",
    "n02231487 walking stick, walkingstick, stick insect",
    "n02233338 cockroach, roach",
    "n02236044 mantis, mantid",
    "n02256656 cicada, cicala",
    "n02259212 leafhopper",
    "n02264363 lacewing, lacewing fly",
    "n02268443 dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    "n02268853 damselfly",
    "n02276258 admiral",
    "n02277742 ringlet, ringlet butterfly",
    "n02279972 monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
    "n02280649 cabbage butterfly",
    "n02281406 sulphur butterfly, sulfur butterfly",
    "n02281787 lycaenid, lycaenid butterfly",
    "n02317335 starfish, sea star",
    "n02319095 sea urchin",
    "n02321529 sea cucumber, holothurian",
    "n02325366 wood rabbit, cottontail, cottontail rabbit",
    "n02326432 hare",
    "n02328150 Angora, Angora rabbit",
    "n02342885 hamster",
    "n02346627 porcupine, hedgehog",
    "n02356798 fox squirrel, eastern fox squirrel, Sciurus niger",
    "n02361337 marmot",
    "n02363005 beaver",
    "n02364673 guinea pig, Cavia cobaya",
    "n02389026 sorrel",
    "n02391049 zebra",
    "n02395406 hog, pig, grunter, squealer, Sus scrofa",
    "n02396427 wild boar, boar, Sus scrofa",
    "n02397096 warthog",
    "n02398521 hippopotamus, hippo, river horse, Hippopotamus amphibius",
    "n02403003 ox",
    "n02408429 water buffalo, water ox, Asiatic buffalo, Bubalus bubalis",
    "n02410509 bison",
    "n02412080 ram, tup",
    "n02415577 bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
    "n02417914 ibex, Capra ibex",
    "n02422106 hartebeest",
    "n02422699 impala, Aepyceros melampus",
    "n02423022 gazelle",
    "n02437312 Arabian camel, dromedary, Camelus dromedarius",
    "n02437616 llama",
    "n02441942 weasel",
    "n02442845 mink",
    "n02443114 polecat, fitch, foulmart, foumart, Mustela putorius",
    "n02443484 black-footed ferret, ferret, Mustela nigripes",
    "n02444819 otter",
    "n02445715 skunk, polecat, wood pussy",
    "n02447366 badger",
    "n02454379 armadillo",
    "n02457408 three-toed sloth, ai, Bradypus tridactylus",
    "n02480495 orangutan, orang, orangutang, Pongo pygmaeus",
    "n02480855 gorilla, Gorilla gorilla",
    "n02481823 chimpanzee, chimp, Pan troglodytes",
    "n02483362 gibbon, Hylobates lar",
    "n02483708 siamang, Hylobates syndactylus, Symphalangus syndactylus",
    "n02484975 guenon, guenon monkey",
    "n02486261 patas, hussar monkey, Erythrocebus patas",
    "n02486410 baboon",
    "n02487347 macaque",
    "n02488291 langur",
    "n02488702 colobus, colobus monkey",
    "n02489166 proboscis monkey, Nasalis larvatus",
    "n02490219 marmoset",
    "n02492035 capuchin, ringtail, Cebus capucinus",
    "n02492660 howler monkey, howler",
    "n02493509 titi, titi monkey",
    "n02493793 spider monkey, Ateles geoffroyi",
    "n02494079 squirrel monkey, Saimiri sciureus",
    "n02497673 Madagascar cat, ring-tailed lemur, Lemur catta",
    "n02500267 indri, indris, Indri indri, Indri brevicaudatus",
    "n02504013 Indian elephant, Elephas maximus",
    "n02504458 African elephant, Loxodonta africana",
    "n02509815 lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
    "n02510455 giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca",
    "n02514041 barracouta, snoek",
    "n02526121 eel",
    "n02536864 coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch",
    "n02606052 rock beauty, Holocanthus tricolor",
    "n02607072 anemone fish",
    "n02640242 sturgeon",
    "n02641379 gar, garfish, garpike, billfish, Lepisosteus osseus",
    "n02643566 lionfish",
    "n02655020 puffer, pufferfish, blowfish, globefish",
    "n02666196 abacus",
    "n02667093 abaya",
    "n02669723 academic gown, academic robe, judge's robe",
    "n02672831 accordion, piano accordion, squeeze box",
    "n02676566 acoustic guitar",
    "n02687172 aircraft carrier, carrier, flattop, attack aircraft carrier",
    "n02690373 airliner",
    "n02692877 airship, dirigible",
    "n02699494 altar",
    "n02701002 ambulance",
    "n02704792 amphibian, amphibious vehicle",
    "n02708093 analog clock",
    "n02727426 apiary, bee house",
    "n02730930 apron",
    "n02747177 ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
    "n02749479 assault rifle, assault gun",
    "n02769748 backpack, back pack, knapsack, packsack, rucksack, haversack",
    "n02776631 bakery, bakeshop, bakehouse",
    "n02777292 balance beam, beam",
    "n02782093 balloon",
    "n02783161 ballpoint, ballpoint pen, ballpen, Biro",
    "n02786058 Band Aid",
    "n02787622 banjo",
    "n02788148 bannister, banister, balustrade, balusters, handrail",
    "n02790996 barbell",
    "n02791124 barber chair",
    "n02791270 barbershop",
    "n02793495 barn",
    "n02794156 barometer",
    "n02795169 barrel, cask",
    "n02797295 barrow, garden cart, lawn cart, wheelbarrow",
    "n02799071 baseball",
    "n02802426 basketball",
    "n02804414 bassinet",
    "n02804610 bassoon",
    "n02807133 bathing cap, swimming cap",
    "n02808304 bath towel",
    "n02808440 bathtub, bathing tub, bath, tub",
    "n02814533 beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
    "n02814860 beacon, lighthouse, beacon light, pharos",
    "n02815834 beaker",
    "n02817516 bearskin, busby, shako",
    "n02823428 beer bottle",
    "n02823750 beer glass",
    "n02825657 bell cote, bell cot",
    "n02834397 bib",
    "n02835271 bicycle-built-for-two, tandem bicycle, tandem",
    "n02837789 bikini, two-piece",
    "n02840245 binder, ring-binder",
    "n02841315 binoculars, field glasses, opera glasses",
    "n02843684 birdhouse",
    "n02859443 boathouse",
    "n02860847 bobsled, bobsleigh, bob",
    "n02865351 bolo tie, bolo, bola tie, bola",
    "n02869837 bonnet, poke bonnet",
    "n02870880 bookcase",
    "n02871525 bookshop, bookstore, bookstall",
    "n02877765 bottlecap",
    "n02879718 bow",
    "n02883205 bow tie, bow-tie, bowtie",
    "n02892201 brass, memorial tablet, plaque",
    "n02892767 brassiere, bra, bandeau",
    "n02894605 breakwater, groin, groyne, mole, bulwark, seawall, jetty",
    "n02895154 breastplate, aegis, egis",
    "n02906734 broom",
    "n02909870 bucket, pail",
    "n02910353 buckle",
    "n02916936 bulletproof vest",
    "n02917067 bullet train, bullet",
    "n02927161 butcher shop, meat market",
    "n02930766 cab, hack, taxi, taxicab",
    "n02939185 caldron, cauldron",
    "n02948072 candle, taper, wax light",
    "n02950826 cannon",
    "n02951358 canoe",
    "n02951585 can opener, tin opener",
    "n02963159 cardigan",
    "n02965783 car mirror",
    "n02966193 carousel, carrousel, merry-go-round, roundabout, whirligig",
    "n02966687 carpenter's kit, tool kit",
    "n02971356 carton",
    "n02974003 car wheel",
    "n02977058 cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM",
    "n02978881 cassette",
    "n02979186 cassette player",
    "n02980441 castle",
    "n02981792 catamaran",
    "n02988304 CD player",
    "n02992211 cello, violoncello",
    "n02992529 cellular telephone, cellular phone, cellphone, cell, mobile phone",
    "n02999410 chain",
    "n03000134 chainlink fence",
    "n03000247 chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour",
    "n03000684 chain saw, chainsaw",
    "n03014705 chest",
    "n03016953 chiffonier, commode",
    "n03017168 chime, bell, gong",
    "n03018349 china cabinet, china closet",
    "n03026506 Christmas stocking",
    "n03028079 church, church building",
    "n03032252 cinema, movie theater, movie theatre, movie house, picture palace",
    "n03041632 cleaver, meat cleaver, chopper",
    "n03042490 cliff dwelling",
    "n03045698 cloak",
    "n03047690 clog, geta, patten, sabot",
    "n03062245 cocktail shaker",
    "n03063599 coffee mug",
    "n03063689 coffeepot",
    "n03065424 coil, spiral, volute, whorl, helix",
    "n03075370 combination lock",
    "n03085013 computer keyboard, keypad",
    "n03089624 confectionery, confectionary, candy store",
    "n03095699 container ship, containership, container vessel",
    "n03100240 convertible",
    "n03109150 corkscrew, bottle screw",
    "n03110669 cornet, horn, trumpet, trump",
    "n03124043 cowboy boot",
    "n03124170 cowboy hat, ten-gallon hat",
    "n03125729 cradle",
    "n03126707 crane",
    "n03127747 crash helmet",
    "n03127925 crate",
    "n03131574 crib, cot",
    "n03133878 Crock Pot",
    "n03134739 croquet ball",
    "n03141823 crutch",
    "n03146219 cuirass",
    "n03160309 dam, dike, dyke",
    "n03179701 desk",
    "n03180011 desktop computer",
    "n03187595 dial telephone, dial phone",
    "n03188531 diaper, nappy, napkin",
    "n03196217 digital clock",
    "n03197337 digital watch",
    "n03201208 dining table, board",
    "n03207743 dishrag, dishcloth",
    "n03207941 dishwasher, dish washer, dishwashing machine",
    "n03208938 disk brake, disc brake",
    "n03216828 dock, dockage, docking facility",
    "n03218198 dogsled, dog sled, dog sleigh",
    "n03220513 dome",
    "n03223299 doormat, welcome mat",
    "n03240683 drilling platform, offshore rig",
    "n03249569 drum, membranophone, tympan",
    "n03250847 drumstick",
    "n03255030 dumbbell",
    "n03259280 Dutch oven",
    "n03271574 electric fan, blower",
    "n03272010 electric guitar",
    "n03272562 electric locomotive",
    "n03290653 entertainment center",
    "n03291819 envelope",
    "n03297495 espresso maker",
    "n03314780 face powder",
    "n03325584 feather boa, boa",
    "n03337140 file, file cabinet, filing cabinet",
    "n03344393 fireboat",
    "n03345487 fire engine, fire truck",
    "n03347037 fire screen, fireguard",
    "n03355925 flagpole, flagstaff",
    "n03372029 flute, transverse flute",
    "n03376595 folding chair",
    "n03379051 football helmet",
    "n03384352 forklift",
    "n03388043 fountain",
    "n03388183 fountain pen",
    "n03388549 four-poster",
    "n03393912 freight car",
    "n03394916 French horn, horn",
    "n03400231 frying pan, frypan, skillet",
    "n03404251 fur coat",
    "n03417042 garbage truck, dustcart",
    "n03424325 gasmask, respirator, gas helmet",
    "n03425413 gas pump, gasoline pump, petrol pump, island dispenser",
    "n03443371 goblet",
    "n03444034 go-kart",
    "n03445777 golf ball",
    "n03445924 golfcart, golf cart",
    "n03447447 gondola",
    "n03447721 gong, tam-tam",
    "n03450230 gown",
    "n03452741 grand piano, grand",
    "n03457902 greenhouse, nursery, glasshouse",
    "n03459775 grille, radiator grille",
    "n03461385 grocery store, grocery, food market, market",
    "n03467068 guillotine",
    "n03476684 hair slide",
    "n03476991 hair spray",
    "n03478589 half track",
    "n03481172 hammer",
    "n03482405 hamper",
    "n03483316 hand blower, blow dryer, blow drier, hair dryer, hair drier",
    "n03485407 hand-held computer, hand-held microcomputer",
    "n03485794 handkerchief, hankie, hanky, hankey",
    "n03492542 hard disc, hard disk, fixed disk",
    "n03494278 harmonica, mouth organ, harp, mouth harp",
    "n03495258 harp",
    "n03496892 harvester, reaper",
    "n03498962 hatchet",
    "n03527444 holster",
    "n03529860 home theater, home theatre",
    "n03530642 honeycomb",
    "n03532672 hook, claw",
    "n03534580 hoopskirt, crinoline",
    "n03535780 horizontal bar, high bar",
    "n03538406 horse cart, horse-cart",
    "n03544143 hourglass",
    "n03584254 iPod",
    "n03584829 iron, smoothing iron",
    "n03590841 jack-o'-lantern",
    "n03594734 jean, blue jean, denim",
    "n03594945 jeep, landrover",
    "n03595614 jersey, T-shirt, tee shirt",
    "n03598930 jigsaw puzzle",
    "n03599486 jinrikisha, ricksha, rickshaw",
    "n03602883 joystick",
    "n03617480 kimono",
    "n03623198 knee pad",
    "n03627232 knot",
    "n03630383 lab coat, laboratory coat",
    "n03633091 ladle",
    "n03637318 lampshade, lamp shade",
    "n03642806 laptop, laptop computer",
    "n03649909 lawn mower, mower",
    "n03657121 lens cap, lens cover",
    "n03658185 letter opener, paper knife, paperknife",
    "n03661043 library",
    "n03662601 lifeboat",
    "n03666591 lighter, light, igniter, ignitor",
    "n03670208 limousine, limo",
    "n03673027 liner, ocean liner",
    "n03676483 lipstick, lip rouge",
    "n03680355 Loafer",
    "n03690938 lotion",
    "n03691459 loudspeaker, speaker, speaker unit, loudspeaker system, speaker system",
    "n03692522 loupe, jeweler's loupe",
    "n03697007 lumbermill, sawmill",
    "n03706229 magnetic compass",
    "n03709823 mailbag, postbag",
    "n03710193 mailbox, letter box",
    "n03710637 maillot",
    "n03710721 maillot, tank suit",
    "n03717622 manhole cover",
    "n03720891 maraca",
    "n03721384 marimba, xylophone",
    "n03724870 mask",
    "n03729826 matchstick",
    "n03733131 maypole",
    "n03733281 maze, labyrinth",
    "n03733805 measuring cup",
    "n03742115 medicine chest, medicine cabinet",
    "n03743016 megalith, megalithic structure",
    "n03759954 microphone, mike",
    "n03761084 microwave, microwave oven",
    "n03763968 military uniform",
    "n03764736 milk can",
    "n03769881 minibus",
    "n03770439 miniskirt, mini",
    "n03770679 minivan",
    "n03773504 missile",
    "n03775071 mitten",
    "n03775546 mixing bowl",
    "n03776460 mobile home, manufactured home",
    "n03777568 Model T",
    "n03777754 modem",
    "n03781244 monastery",
    "n03782006 monitor",
    "n03785016 moped",
    "n03786901 mortar",
    "n03787032 mortarboard",
    "n03788195 mosque",
    "n03788365 mosquito net",
    "n03791053 motor scooter, scooter",
    "n03792782 mountain bike, all-terrain bike, off-roader",
    "n03792972 mountain tent",
    "n03793489 mouse, computer mouse",
    "n03794056 mousetrap",
    "n03796401 moving van",
    "n03803284 muzzle",
    "n03804744 nail",
    "n03814639 neck brace",
    "n03814906 necklace",
    "n03825788 nipple",
    "n03832673 notebook, notebook computer",
    "n03837869 obelisk",
    "n03838899 oboe, hautboy, hautbois",
    "n03840681 ocarina, sweet potato",
    "n03841143 odometer, hodometer, mileometer, milometer",
    "n03843555 oil filter",
    "n03854065 organ, pipe organ",
    "n03857828 oscilloscope, scope, cathode-ray oscilloscope, CRO",
    "n03866082 overskirt",
    "n03868242 oxcart",
    "n03868863 oxygen mask",
    "n03871628 packet",
    "n03873416 paddle, boat paddle",
    "n03874293 paddlewheel, paddle wheel",
    "n03874599 padlock",
    "n03876231 paintbrush",
    "n03877472 pajama, pyjama, pj's, jammies",
    "n03877845 palace",
    "n03884397 panpipe, pandean pipe, syrinx",
    "n03887697 paper towel",
    "n03888257 parachute, chute",
    "n03888605 parallel bars, bars",
    "n03891251 park bench",
    "n03891332 parking meter",
    "n03895866 passenger car, coach, carriage",
    "n03899768 patio, terrace",
    "n03902125 pay-phone, pay-station",
    "n03903868 pedestal, plinth, footstall",
    "n03908618 pencil box, pencil case",
    "n03908714 pencil sharpener",
    "n03916031 perfume, essence",
    "n03920288 Petri dish",
    "n03924679 photocopier",
    "n03929660 pick, plectrum, plectron",
    "n03929855 pickelhaube",
    "n03930313 picket fence, paling",
    "n03930630 pickup, pickup truck",
    "n03933933 pier",
    "n03935335 piggy bank, penny bank",
    "n03937543 pill bottle",
    "n03938244 pillow",
    "n03942813 ping-pong ball",
    "n03944341 pinwheel",
    "n03947888 pirate, pirate ship",
    "n03950228 pitcher, ewer",
    "n03954731 plane, carpenter's plane, woodworking plane",
    "n03956157 planetarium",
    "n03958227 plastic bag",
    "n03961711 plate rack",
    "n03967562 plow, plough",
    "n03970156 plunger, plumber's helper",
    "n03976467 Polaroid camera, Polaroid Land camera",
    "n03976657 pole",
    "n03977966 police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
    "n03980874 poncho",
    "n03982430 pool table, billiard table, snooker table",
    "n03983396 pop bottle, soda bottle",
    "n03991062 pot, flowerpot",
    "n03992509 potter's wheel",
    "n03995372 power drill",
    "n03998194 prayer rug, prayer mat",
    "n04004767 printer",
    "n04005630 prison, prison house",
    "n04008634 projectile, missile",
    "n04009552 projector",
    "n04019541 puck, hockey puck",
    "n04023962 punching bag, punch bag, punching ball, punchball",
    "n04026417 purse",
    "n04033901 quill, quill pen",
    "n04033995 quilt, comforter, comfort, puff",
    "n04037443 racer, race car, racing car",
    "n04039381 racket, racquet",
    "n04040759 radiator",
    "n04041544 radio, wireless",
    "n04044716 radio telescope, radio reflector",
    "n04049303 rain barrel",
    "n04065272 recreational vehicle, RV, R.V.",
    "n04067472 reel",
    "n04069434 reflex camera",
    "n04070727 refrigerator, icebox",
    "n04074963 remote control, remote",
    "n04081281 restaurant, eating house, eating place, eatery",
    "n04086273 revolver, six-gun, six-shooter",
    "n04090263 rifle",
    "n04099969 rocking chair, rocker",
    "n04111531 rotisserie",
    "n04116512 rubber eraser, rubber, pencil eraser",
    "n04118538 rugby ball",
    "n04118776 rule, ruler",
    "n04120489 running shoe",
    "n04125021 safe",
    "n04127249 safety pin",
    "n04131690 saltshaker, salt shaker",
    "n04133789 sandal",
    "n04136333 sarong",
    "n04141076 sax, saxophone",
    "n04141327 scabbard",
    "n04141975 scale, weighing machine",
    "n04146614 school bus",
    "n04147183 schooner",
    "n04149813 scoreboard",
    "n04152593 screen, CRT screen",
    "n04153751 screw",
    "n04154565 screwdriver",
    "n04162706 seat belt, seatbelt",
    "n04179913 sewing machine",
    "n04192698 shield, buckler",
    "n04200800 shoe shop, shoe-shop, shoe store",
    "n04201297 shoji",
    "n04204238 shopping basket",
    "n04204347 shopping cart",
    "n04208210 shovel",
    "n04209133 shower cap",
    "n04209239 shower curtain",
    "n04228054 ski",
    "n04229816 ski mask",
    "n04235860 sleeping bag",
    "n04238763 slide rule, slipstick",
    "n04239074 sliding door",
    "n04243546 slot, one-armed bandit",
    "n04251144 snorkel",
    "n04252077 snowmobile",
    "n04252225 snowplow, snowplough",
    "n04254120 soap dispenser",
    "n04254680 soccer ball",
    "n04254777 sock",
    "n04258138 solar dish, solar collector, solar furnace",
    "n04259630 sombrero",
    "n04263257 soup bowl",
    "n04264628 space bar",
    "n04265275 space heater",
    "n04266014 space shuttle",
    "n04270147 spatula",
    "n04273569 speedboat",
    "n04275548 spider web, spider's web",
    "n04277352 spindle",
    "n04285008 sports car, sport car",
    "n04286575 spotlight, spot",
    "n04296562 stage",
    "n04310018 steam locomotive",
    "n04311004 steel arch bridge",
    "n04311174 steel drum",
    "n04317175 stethoscope",
    "n04325704 stole",
    "n04326547 stone wall",
    "n04328186 stopwatch, stop watch",
    "n04330267 stove",
    "n04332243 strainer",
    "n04335435 streetcar, tram, tramcar, trolley, trolley car",
    "n04336792 stretcher",
    "n04344873 studio couch, day bed",
    "n04346328 stupa, tope",
    "n04347754 submarine, pigboat, sub, U-boat",
    "n04350905 suit, suit of clothes",
    "n04355338 sundial",
    "n04355933 sunglass",
    "n04356056 sunglasses, dark glasses, shades",
    "n04357314 sunscreen, sunblock, sun blocker",
    "n04366367 suspension bridge",
    "n04367480 swab, swob, mop",
    "n04370456 sweatshirt",
    "n04371430 swimming trunks, bathing trunks",
    "n04371774 swing",
    "n04372370 switch, electric switch, electrical switch",
    "n04376876 syringe",
    "n04380533 table lamp",
    "n04389033 tank, army tank, armored combat vehicle, armoured combat vehicle",
    "n04392985 tape player",
    "n04398044 teapot",
    "n04399382 teddy, teddy bear",
    "n04404412 television, television system",
    "n04409515 tennis ball",
    "n04417672 thatch, thatched roof",
    "n04418357 theater curtain, theatre curtain",
    "n04423845 thimble",
    "n04428191 thresher, thrasher, threshing machine",
    "n04429376 throne",
    "n04435653 tile roof",
    "n04442312 toaster",
    "n04443257 tobacco shop, tobacconist shop, tobacconist",
    "n04447861 toilet seat",
    "n04456115 torch",
    "n04458633 totem pole",
    "n04461696 tow truck, tow car, wrecker",
    "n04462240 toyshop",
    "n04465501 tractor",
    "n04467665 trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi",
    "n04476259 tray",
    "n04479046 trench coat",
    "n04482393 tricycle, trike, velocipede",
    "n04483307 trimaran",
    "n04485082 tripod",
    "n04486054 triumphal arch",
    "n04487081 trolleybus, trolley coach, trackless trolley",
    "n04487394 trombone",
    "n04493381 tub, vat",
    "n04501370 turnstile",
    "n04505470 typewriter keyboard",
    "n04507155 umbrella",
    "n04509417 unicycle, monocycle",
    "n04515003 upright, upright piano",
    "n04517823 vacuum, vacuum cleaner",
    "n04522168 vase",
    "n04523525 vault",
    "n04525038 velvet",
    "n04525305 vending machine",
    "n04532106 vestment",
    "n04532670 viaduct",
    "n04536866 violin, fiddle",
    "n04540053 volleyball",
    "n04542943 waffle iron",
    "n04548280 wall clock",
    "n04548362 wallet, billfold, notecase, pocketbook",
    "n04550184 wardrobe, closet, press",
    "n04552348 warplane, military plane",
    "n04553703 washbasin, handbasin, washbowl, lavabo, wash-hand basin",
    "n04554684 washer, automatic washer, washing machine",
    "n04557648 water bottle",
    "n04560804 water jug",
    "n04562935 water tower",
    "n04579145 whiskey jug",
    "n04579432 whistle",
    "n04584207 wig",
    "n04589890 window screen",
    "n04590129 window shade",
    "n04591157 Windsor tie",
    "n04591713 wine bottle",
    "n04592741 wing",
    "n04596742 wok",
    "n04597913 wooden spoon",
    "n04599235 wool, woolen, woollen",
    "n04604644 worm fence, snake fence, snake-rail fence, Virginia fence",
    "n04606251 wreck",
    "n04612504 yawl",
    "n04613696 yurt",
    "n06359193 web site, website, internet site, site",
    "n06596364 comic book",
    "n06785654 crossword puzzle, crossword",
    "n06794110 street sign",
    "n06874185 traffic light, traffic signal, stoplight",
    "n07248320 book jacket, dust cover, dust jacket, dust wrapper",
    "n07565083 menu",
    "n07579787 plate",
    "n07583066 guacamole",
    "n07584110 consomme",
    "n07590611 hot pot, hotpot",
    "n07613480 trifle",
    "n07614500 ice cream, icecream",
    "n07615774 ice lolly, lolly, lollipop, popsicle",
    "n07684084 French loaf",
    "n07693725 bagel, beigel",
    "n07695742 pretzel",
    "n07697313 cheeseburger",
    "n07697537 hotdog, hot dog, red hot",
    "n07711569 mashed potato",
    "n07714571 head cabbage",
    "n07714990 broccoli",
    "n07715103 cauliflower",
    "n07716358 zucchini, courgette",
    "n07716906 spaghetti squash",
    "n07717410 acorn squash",
    "n07717556 butternut squash",
    "n07718472 cucumber, cuke",
    "n07718747 artichoke, globe artichoke",
    "n07720875 bell pepper",
    "n07730033 cardoon",
    "n07734744 mushroom",
    "n07742313 Granny Smith",
    "n07745940 strawberry",
    "n07747607 orange",
    "n07749582 lemon",
    "n07753113 fig",
    "n07753275 pineapple, ananas",
    "n07753592 banana",
    "n07754684 jackfruit, jak, jack",
    "n07760859 custard apple",
    "n07768694 pomegranate",
    "n07802026 hay",
    "n07831146 carbonara",
    "n07836838 chocolate sauce, chocolate syrup",
    "n07860988 dough",
    "n07871810 meat loaf, meatloaf",
    "n07873807 pizza, pizza pie",
    "n07875152 potpie",
    "n07880968 burrito",
    "n07892512 red wine",
    "n07920052 espresso",
    "n07930864 cup",
    "n07932039 eggnog",
    "n09193705 alp",
    "n09229709 bubble",
    "n09246464 cliff, drop, drop-off",
    "n09256479 coral reef",
    "n09288635 geyser",
    "n09332890 lakeside, lakeshore",
    "n09399592 promontory, headland, head, foreland",
    "n09421951 sandbar, sand bar",
    "n09428293 seashore, coast, seacoast, sea-coast",
    "n09468604 valley, vale",
    "n09472597 volcano",
    "n09835506 ballplayer, baseball player",
    "n10148035 groom, bridegroom",
    "n10565667 scuba diver",
    "n11879895 rapeseed",
    "n11939491 daisy",
    "n12057211 yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
    "n12144580 corn",
    "n12267677 acorn",
    "n12620546 hip, rose hip, rosehip",
    "n12768682 buckeye, horse chestnut, conker",
    "n12985857 coral fungus",
    "n12998815 agaric",
    "n13037406 gyromitra",
    "n13040303 stinkhorn, carrion fungus",
    "n13044778 earthstar",
    "n13052670 hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa",
    "n13054560 bolete",
    "n13133613 ear, spike, capitulum",
    "n15075141 toilet tissue, toilet paper, bathroom tissue",
  ]
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
