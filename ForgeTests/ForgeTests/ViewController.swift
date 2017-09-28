import UIKit
import Metal
import MetalPerformanceShaders

/*
  Since we can't use XCTest, we'll do a very primitive form of unit testing.

  You have to create the test classes and call each of the test cases by hand.
  Any output goes to stdout.
  As soon as something trips an assert, the app terminates.
  You need to run this app on a device.

  But even this primitive approach is good enough for making sure the compute
  kernels work properly etc.
*/

var device: MTLDevice!
var commandQueue: MTLCommandQueue!

class ViewController: UIViewController {
  @IBOutlet weak var button: UIButton!

  override func viewDidLoad() {
    super.viewDidLoad()

    device = MTLCreateSystemDefaultDevice()
    if device == nil {
      print("Error: this device does not support Metal")
      return
    }

    guard MPSSupportsMTLDevice(device) else {
      print("Error: this device does not support Metal Performance Shaders")
      return
    }

    commandQueue = device.makeCommandQueue()

    // Wait a second for the app to finish loading before running the tests.
    DispatchQueue.main.asyncAfter(deadline: .now() + 1, execute: runTests)
  }

  @IBAction func runTests() {
    print("\n-----Running Tests-----")
    verbose = true
    button.isEnabled = false

    let arrayTests = ArrayTests()
    arrayTests.testArgmax()
    arrayTests.testArgsort()
    arrayTests.testGather()
    
    arrayTests.testMakeArray()
    arrayTests.testReshape()
    arrayTests.testTranspose()
    arrayTests.testSlice()

    let layerTests = LayerTests()
    
    layerTests.testMPSImages()
    layerTests.testActivationLayer()
    layerTests.testSpaceToDepthX2Layer()
    layerTests.testCollect()
    layerTests.testSimpleMerge()
    

    layerTests.testComplexMerge()
    layerTests.testZeroPaddingLayer_1ch()
    layerTests.testZeroPaddingLayer_3ch()
    //layerTests.testZeroPaddingLayer_3ch_Image()
 
    layerTests.testSubtractMean()
    /*
    layerTests.testVGG16(debug: true)
    layerTests.testResNet(debug: true)
    layerTests.testInception_V3(debug: true)
    layerTests.testInceptionResnet(debug: true)
 */
    let modeTests = ModelTests()
    modeTests.testVGG16(debug: false)
    modeTests.testResnet50(debug: false)
    modeTests.testInceptionV3(debug: false)
    modeTests.testInceptionResnetV2(debug: false)

/*
    let basicConvTests = BasicConvolutionTests()
    basicConvTests.testCorrectness()

    let depthwiseConvTests = DepthwiseConvolutionTests()
    depthwiseConvTests.testCorrectness(useMPSDepthWise: false)
    depthwiseConvTests.testCorrectness(useMPSDepthWise: true)
    //depthwiseConvTests.testGroups()
*/
    print("All tests successful!")
    button.isEnabled = true
  }
}
