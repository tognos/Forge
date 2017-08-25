import UIKit
import Metal
import MetalPerformanceShaders
import AVFoundation
import CoreMedia
import Forge

let TinyMaxBuffersInFlight = 3   // use triple buffering
let YoloMaxBuffersInFlight = 1   // use single buffering for full yolo

var TinyYolo = true

// The labels for the 20 classes.
let voc_labels = [
  "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
  "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
  "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

let coco_labels = [
  "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
  "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
  "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
  "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

extension UIImage {
  class func image(from layer: CALayer) -> UIImage? {
    
    UIGraphicsBeginImageContextWithOptions(layer.bounds.size,
                                           layer.isOpaque, UIScreen.main.scale)
    
    // Don't proceed unless we have context
    guard let context = UIGraphicsGetCurrentContext() else {
      UIGraphicsEndImageContext()
      return nil
    }
    
    layer.render(in: context)
    let image = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    
    return image
  }
}

public func makeImage(size: CGSize, from: CALayer) -> UIImage? {
  let bounds = CGRect(origin: .zero, size: size)
  let colorSpace = CGColorSpaceCreateDeviceRGB()
  let (width, height) = (Int(size.width), Int(size.height))
  
  // Build Core Graphics ARGB context
  guard let context = CGContext(data: nil, width: width,
                                height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue)
    else { return nil }
  print("created context")
  // Prepare CG Context for UIKit
  UIGraphicsPushContext(context); defer { UIGraphicsPopContext() }
  
  // Draw to context using UIKit calls
  from.render(in: context)
  print("rendered")
  
  // Fetch the image from the context
  guard let imageRef = context.makeImage() else { return nil }
  print("made image")
  return UIImage(cgImage: imageRef, scale: 1.0, orientation: UIImageOrientation.downMirrored)
  //return UIImage(cgImage: imageRef)
}

public func makeImage(size: CGSize, from: CGLayer) -> UIImage? {
  let bounds = CGRect(origin: .zero, size: size)
  let colorSpace = CGColorSpaceCreateDeviceRGB()
  let (width, height) = (Int(size.width), Int(size.height))
  
  // Build Core Graphics ARGB context
  guard let context = CGContext(data: nil, width: width,
                                height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue)
    else { return nil }
  
  // Prepare CG Context for UIKit
  UIGraphicsPushContext(context); defer { UIGraphicsPopContext() }
  
  // Draw to context using UIKit calls
  let destination = CGRect(origin: .zero, size: size)
  context.draw(from, in: destination)
  
  // Fetch the image from the context
  guard let imageRef = context.makeImage() else { return nil }
  return UIImage(cgImage: imageRef)
}


extension UIImage {
  func saveAsDocument(fileName: String) {
    print("Saving", fileName)
    let documentsDirectoryURL = try! FileManager().url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    // create a name for your image
    let fileURL = documentsDirectoryURL.appendingPathComponent(fileName)
    
    do {
      try UIImageJPEGRepresentation(self,1.0)!.write(to: fileURL)
      print("Image added successfully")
    } catch {
      print(error)
    }
  }
}

func saveAsRawFile(fileName: String, data : Data) {
  print("Saving data as ", fileName)
  let documentsDirectoryURL = try! FileManager().url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
  // create a name for your image
  let fileURL = documentsDirectoryURL.appendingPathComponent(fileName)
  
  do {
    try data.write(to: fileURL)
    print("Data saved successfully")
  } catch {
    print(error)
  }
}

class CameraViewController: UIViewController {
  
  @IBOutlet weak var videoPreview: UIView!
  @IBOutlet weak var timeLabel: UILabel!
  @IBOutlet weak var debugImageView: UIImageView!
  @IBOutlet weak var videoPostview: UIImageView!

  var videoCapture: VideoCapture!
  var device: MTLDevice!
  
  var tinyCommandQueue: MTLCommandQueue!
  var yoloCommandQueue: MTLCommandQueue!
  var activeCommandQueue: MTLCommandQueue!
  
  var tinyRunner: Runner!
  var yoloRunner: Runner!
  var activeRunner: Runner!
  
  var tinyNetwork: YOLO!
  var yoloNetwork: YOLO!
  var activeNetwork: YOLO!
  
  var startupGroup = DispatchGroup()
  
  var boundingBoxes = [BoundingBox]()
  var offScreen: CALayer!
  
  var tinyColors: [UIColor] = []
  var yoloColors: [UIColor] = []

  var sBounds : CGRect?
  
  var viewBounds : CGRect {
    if let result = sBounds {
      return result
    } else {
      DispatchQueue.main.sync {
        sBounds = view.bounds
      }
      return sBounds!
    }
  }
  /*
   override var supportedInterfaceOrientations:UIInterfaceOrientationMask {
   return UIInterfaceOrientationMask.landscapeLeft
   }
   */
  override func viewDidLoad() {
    super.viewDidLoad()
    
    timeLabel.text = ""
    
    device = MTLCreateSystemDefaultDevice()
    if device == nil {
      print("Error: this device does not support Metal")
      return
    }
    
    tinyCommandQueue = device.makeCommandQueue()
    yoloCommandQueue = device.makeCommandQueue()

    // The app can show up to 10 detections at a time. You can increase this
    // limit by allocating more BoundingBox objects, but there's only so much
    // room on the screen. (You also need to change the limit in YOLO.swift.)
    for _ in 0..<10 {
      boundingBoxes.append(BoundingBox())
    }
    
    // Make colors for the bounding boxes. There is one color for each class,
    // 20 classes in total.
    for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
      for g: CGFloat in [0.2, 0.7] {
        for b: CGFloat in [0.4, 0.8] {
          let color = UIColor(red: r, green: g, blue: b, alpha: 1)
          tinyColors.append(color)
        }
      }
    }
    
    // Make colors for the bounding boxes. There is one color for each class,
    // 80 classes in total.
    for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
      for g: CGFloat in [0.4, 0.6, 0.8, 1.0] {
        for b: CGFloat in [0.4, 0.6, 0.8, 1.0] {
          let color = UIColor(red: r, green: g, blue: b, alpha: 1)
          yoloColors.append(color)
        }
      }
    }
    
    videoCapture = VideoCapture(device: device)
    videoCapture.delegate = self
    //videoCapture.fps = 5
    videoCapture.fps = 20
    
    // Initialize the camera.
    startupGroup.enter()
    
    //  let preset = AVCaptureSession.Preset.vga640x480
    //  let orientation = AVCaptureVideoOrientation.portrait

    let preset = AVCaptureSession.Preset.hd1280x720
    let orientation = AVCaptureVideoOrientation.landscapeRight

    videoCapture.setUp(sessionPreset: preset, orientation: orientation) { success in
      // Add the video preview into the UI.
      if let previewLayer = self.videoCapture.previewLayer {
        self.videoPreview.layer.addSublayer(previewLayer)
        self.resizePreviewLayer()
      }
      self.startupGroup.leave()
    }
    
    // Initialize the neural network.
    startupGroup.enter()
    createNeuralNetwork {
      self.startupGroup.leave()
    }
    
    startupGroup.notify(queue: .main) {
      // Add the bounding box layers to the UI, on top of the video preview.
      #if true
        // Once the NN is set up, we can start capturing live video.
        for box in self.boundingBoxes {
          box.addToLayer(self.videoPreview.layer)
          //box.addToLayer(self.videoPostview.layer)
        }
        self.videoCapture.start()
      #else
        //self.offScreen = CALayer()
        self.offScreen = self.videoPostview.layer
        for box in self.boundingBoxes {
          box.addToLayer(self.offScreen)
        }
        DispatchQueue.global(qos: .userInitiated).async {
          self.test_on_images()
        }
      #endif
    }
  }
  
  func test_on_images() {
    var textures = [MTLTexture]()
    let names = ["dog.jpg","eagle.jpg","giraffe.jpg","horses.jpg","person.jpg","scream.jpg","bricks.png","danger.png","gray.png"]
    for name in names {
      if let texture = loadTexture(named: name) {
        textures.append(texture)
      } else {
        print("Can't load image:",name)
      }
    }
    
    for (index, texture) in textures.enumerated() {
      let targetBounds = CGRect(x: 0, y: 0, width: texture.width, height: texture.height)
      print("Predict on:"+names[index])
      DispatchQueue.main.sync {
        offScreen.bounds = targetBounds
        //offScreen.contentsRect = targetBounds
        //offScreen.masksToBounds = true
        self.offScreen.layoutSublayers()
      }
      DispatchQueue.main.sync {
        let timg = UIImage.image(texture: texture)
        //offScreen.contents = timg.cgImage
        self.videoPostview.image = timg
        self.offScreen.layoutSublayers()
        print("Offscreen set to:"+names[index])
      }
      activeRunner.predict(network: activeNetwork, texture: texture, queue: DispatchQueue.main) { result in
        let timg = UIImage.image(texture: texture)
        //self.videoPreview.layer.contents = timg.cgImage
        self.videoPostview.image = timg
        self.debugImageView.layer.contents = timg.cgImage
        
        self.show(predictions: result.predictions,
                  srcImageWidth: CGFloat(texture.width),
                  srcImageHeight: CGFloat(texture.height),
                  targetBounds: targetBounds)

        print("Predicted on:"+names[index])
        //let outImage = UIImage.image(from: offScreen)
        let size = CGSize(width: texture.width, height: texture.height)
        let outImage = makeImage(size: size, from: self.offScreen)
        if let description = outImage?.size.debugDescription {
          print("Snapshotted:"+names[index], "size", description)
        }
        outImage?.saveAsDocument(fileName: names[index])
        result.debugLayer?.withUnsafeBytes({ (bufferptr) in
          let data = Data(bufferptr)
          saveAsRawFile(fileName: "result-"+names[index]+".floats", data: data)
        })
      }
    }
  }
  
  override func didReceiveMemoryWarning() {
    super.didReceiveMemoryWarning()
    print(#function)
  }
  
  // MARK: - UI stuff
  
  override func viewWillLayoutSubviews() {
    super.viewWillLayoutSubviews()
    resizePreviewLayer()
  }
  
  override var preferredStatusBarStyle: UIStatusBarStyle {
    return .lightContent
  }
  
  func resizePreviewLayer() {
    videoCapture.previewLayer?.frame = videoPreview.bounds
  }
  
  func setNetwork(tiny: Bool) {
    if (tiny) {
      activeNetwork = tinyNetwork
      activeRunner = tinyRunner
      activeCommandQueue = tinyCommandQueue
    } else {
      activeNetwork = yoloNetwork
      activeRunner = yoloRunner
      activeCommandQueue = yoloCommandQueue
    }
  }
  
  // MARK: - Neural network
  
  func createNeuralNetwork(completion: @escaping () -> Void) {
    // Make sure the current device supports MetalPerformanceShaders.
    guard MPSSupportsMTLDevice(device) else {
      print("Error: this device does not support Metal Performance Shaders")
      return
    }
    
    yoloRunner = Runner(commandQueue: yoloCommandQueue, inflightBuffers: 1)
    tinyRunner = Runner(commandQueue: tinyCommandQueue, inflightBuffers: 3)

    // Because it may take a few seconds to load the network's parameters,
    // perform the construction of the neural network in the background.
    DispatchQueue.global().async {
      
      timeIt("Setting up neural network") {
        self.tinyNetwork = TinyYOLO(device: self.device, inflightBuffers: 3)
        self.yoloNetwork = YOLO2(device: self.device, inflightBuffers: 1)
        self.setNetwork(tiny: true)
      }
      
      DispatchQueue.main.async(execute: completion)
    }
  }
  
  func predict(texture: MTLTexture, targetBounds: CGRect) {
    // Since we want to run in "realtime", every call to predict() results in
    // a UI update on the main thread. It would be a waste to make the neural
    // network do work and then immediately throw those results away, so the 
    // network should not be called more often than the UI thread can handle.
    // It is up to VideoCapture to throttle how often the neural network runs.
    
    activeRunner.predict(network: activeNetwork, texture: texture, queue: .main) { result in
//      let timg = UIImage.image(texture: texture)
//      self.videoPostview.image = timg
//      self.debugImageView.layer.contents = timg.cgImage
      
      self.show(predictions: result.predictions,
                srcImageWidth: CGFloat(texture.width),
                srcImageHeight: CGFloat(texture.height),
                targetBounds: targetBounds)
      
      if let texture = result.debugTexture {
        self.debugImageView.image = UIImage.image(texture: texture)
      }
      self.timeLabel.text = String(format: "Elapsed %.5f seconds (%.2f FPS)", result.elapsed, 1/result.elapsed)
    }
  }
  
  private func show(predictions: [YOLO.Prediction], srcImageWidth: CGFloat, srcImageHeight: CGFloat, targetBounds: CGRect ) {
    for i in 0..<boundingBoxes.count {
      if i < predictions.count {
        let prediction = predictions[i]
        
        // The predicted bounding box is in the coordinate space of the input
        // image, which is a square image of 416x416 or 608x608 pixels. We want to show it
        // on the video preview, which is as wide as the screen and has a 4:3
        // aspect ratio. The video preview also may be letterboxed at the top
        // and bottom.
        //let width = view.bounds.width
        // let height = width * 720 / 1280
        // let scaleX = width / 608
        // let scaleY = height / 608
        //print("srcImageWidth=", srcImageWidth, "srcImageHeight=",srcImageHeight,
        //      "network.inputWidth=",network.inputWidth,"network.inputHeight=", network.inputHeight)
        let width = targetBounds.width
        let height = width * srcImageHeight / srcImageWidth
        let scaleX = width / CGFloat(activeNetwork.inputWidth)
        let scaleY = height / CGFloat(activeNetwork.inputHeight)
        
        let left = (targetBounds.width - width) / 2
        let top = (targetBounds.height - height) / 2
        
        // Translate and scale the rectangle to our own coordinate system.
        var rect = prediction.rect
        rect.origin.x *= scaleX
        rect.origin.y *= scaleY
        rect.origin.x += left
        rect.origin.y += top
        rect.size.width *= scaleX
        rect.size.height *= scaleY
        
        let labels = activeNetwork === tinyNetwork ? voc_labels : coco_labels
        let colors = activeNetwork === tinyNetwork ? tinyColors : yoloColors

        // Show the bounding box.
        let label = String(format: "%@ %.1f", labels[prediction.classIndex], prediction.score * 100)
        let color = colors[prediction.classIndex]
        boundingBoxes[i].show(frame: rect, label: label, color: color)
        
      } else {
        boundingBoxes[i].hide()
      }
    }
  }
  
  private func show_new(predictions: [YOLO.Prediction], srcImageWidth: CGFloat, srcImageHeight: CGFloat, targetBounds: CGRect ) {
    for i in 0..<boundingBoxes.count {
      if i < predictions.count {
        let prediction = predictions[i]
        
        let width = targetBounds.width
        let height = width * srcImageHeight / srcImageWidth
        
        let left = (targetBounds.width - width) / 2
        let top = (targetBounds.height - height) / 2
        
        // Translate and scale the rectangle to our own coordinate system.
        var rect = prediction.rect
        rect.origin.x *= width
        rect.origin.y *= height
        rect.origin.x += left
        rect.origin.y += top
        rect.size.width *= width
        rect.size.height *= height
        
        let labels = activeNetwork === tinyNetwork ? voc_labels : coco_labels
        let colors = activeNetwork === tinyNetwork ? tinyColors : yoloColors

        // Show the bounding box.
        let label = String(format: "%@ %.1f", labels[prediction.classIndex], prediction.score * 100)
        let color = colors[prediction.classIndex]
        boundingBoxes[i].show(frame: rect, label: label, color: color)
        
      } else {
        boundingBoxes[i].hide()
      }
    }
  }
}
extension CameraViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoTexture texture: MTLTexture?, timestamp: CMTime) {
    // Call the predict() method, which encodes the neural net's GPU commands,
    // on our own thread. Since NeuralNetwork.predict() can block, so can our
    // thread. That is OK, since any new frames will be automatically dropped
    // while the serial dispatch queue is blocked.
    if let texture = texture {
      predict(texture: texture, targetBounds: self.viewBounds)
    }
  }
  
  func videoCapture(_ capture: VideoCapture, didCapturePhotoTexture texture: MTLTexture?, previewImage: UIImage?) {
    // not implemented
  }
}
