//
//  ModelTests.swift
//  ForgeTests
//
//  Created by Pavel Mayer on 28.09.17.
//  Copyright © 2017 Tognos GmbH. All rights reserved.
//

import Foundation
import Forge
import MetalPerformanceShaders

class ModelTests {
  func testVGG16(debug: Bool) {
    print("\(self).\(#function)")
    testModel(net: Vgg16Builder(device: device), testImageName: "final1-224.jpg",
              correctTop: [717, 661],
              debug: debug)
  }
  func testResnet50(debug: Bool) {
    print("\(self).\(#function)")
    testModel(net: Resnet50Builder(device: device), testImageName: "final1-224.jpg",
              correctTop: [717,661],
              debug: debug)
  }
  func testInceptionV3(debug: Bool) {
    print("\(self).\(#function)")
    testModel(net: InceptionV3Builder(device: device), testImageName: "final1-299.jpg",
              correctTop: [717, 661],
              debug: debug)
  }
  func testInceptionResnetV2(debug: Bool) {
    print("\(self).\(#function)")
    testModel(net: InceptionResnetV2Builder(device: device), testImageName: "final1-299.jpg",
              correctTop: [717, 864],
              debug: debug)
  }
  func testMobileNet(debug: Bool) {
    print("\(self).\(#function)")
    testModel(net: MobilenetBuilder(device: device), testImageName: "final1-224.jpg",
              correctTop: [717, 661],
              debug: debug)
  }
  
  func testModel(net : NetworkBuilder, testImageName: String, correctTop: [Int], debug: Bool) {
    print("\(self).\(#function) \(net.name)")
    
    let success = net.compile(inflightBuffers: 1)
    precondition(success)
    print("Successfully compiled model \(net.name)")
    let model = net.model
    let testInputImage = loadTexture(named: testImageName)
    
    var probabilitiesImage : MPSImage
    
    if debug {
      print(model.summary())
      print(model.debugGraph())
      let dumper = DocumentDirectoryDumper(filePrefix: net.name)
      model.debugEvaluate(commandQueue: commandQueue, device: device, input: nil, inputTexure: testInputImage!, dumper: dumper)
      if !dumper.shapes.saveAsJson(fileName: "shapes-"+net.name+".json") {
        print("Error dumping shapes")
      }
      probabilitiesImage = model.outputImage(inflightIndex: 0)
    } else {
      probabilitiesImage = model.evaluate(commandQueue: commandQueue, device: device, input: nil, inputTexure: testInputImage!)
    }
    let probabilities = probabilitiesImage.toFloatArray()

    typealias Prediction = (label: String, probability: Float, index: Int)
    var result = NeuralNetworkResult<Prediction>()
    result.predictions = probabilities.top(k: correctTop.count).map { x -> Prediction in (imageNetLabels[x.0], x.1, x.0) }
    print("predictions:\(result.predictions)")
    for (predicted, correct) in zip(result.predictions, correctTop) {
      precondition(predicted.index == correct, "Bad prediction: \(predicted)")
    }
    precondition(result.predictions[0].probability > 0.5, "confidence < 0.5 for top prediction")
  }
}
