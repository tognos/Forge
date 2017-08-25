import MetalPerformanceShaders
import Forge

/*
 The tiny-yolo-voc network from YOLOv2. https://pjreddie.com/darknet/yolo/
 
 This implementation is cobbled together from the following sources:
 
 - https://github.com/pjreddie/darknet
 - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowYoloDetector.java
 - https://github.com/allanzelener/YAD2K
 */

class YOLO  : NeuralNetwork {
  
  typealias PredictionType = YOLO.Prediction
  
  struct Prediction {
    let classIndex: Int
    let score: Float
    let rect: CGRect
  }
  
  let anchors: [Float]
  
  let model: Model
  let inputWidth : Int
  let inputHeight: Int
  
  let blockSize: Int
  let gridHeight: Int
  let gridWidth: Int
  let boxesPerCell: Int
  let numClasses: Int
  let threshold: Float
  
  var debugOut : Tensor?

  public init(model: Model,
              inputWidth : Int, inputHeight: Int,
              blockSize: Int,
              gridHeight: Int, gridWidth: Int,
              boxesPerCell: Int,
              numClasses: Int,
              anchors : [Float],
              threshold: Float
    ) {
    self.model = model
    precondition(inputWidth / blockSize == gridWidth)
    precondition(inputHeight / blockSize == gridHeight)
    self.inputWidth = inputWidth
    self.inputHeight = inputHeight
    self.blockSize = blockSize
    self.gridWidth = gridWidth
    self.gridHeight = gridHeight
    self.boxesPerCell = boxesPerCell
    self.numClasses = numClasses
    self.anchors = anchors
    self.threshold = threshold
  }

  public func encode(commandBuffer: MTLCommandBuffer, texture sourceTexture: MTLTexture, inflightIndex: Int) {
    model.encode(commandBuffer: commandBuffer, texture: sourceTexture, inflightIndex: inflightIndex)
  }
  public func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<Prediction> {

    
    let featuresImage = model.outputImage(inflightIndex: inflightIndex)
    let features = featuresImage.toFloatArray()
    
    var predictions = [Prediction]()
    
    // This helper function finds the offset in the features array for a given
    // channel for a particular pixel. (See the comment below.)
    func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
      let slice = channel / 4
      let indexInSlice = channel - slice*4
      let offset = slice*gridHeight*gridWidth*4 + y*gridWidth*4 + x*4 + indexInSlice
      return offset
    }
    
    // The 416x416 (608x608) image is divided into a 13x13 (19x19) grid. Each of these grid cells
    // will predict 5 bounding boxes (boxesPerCell). A bounding box consists of 
    // five data items: x, y, width, height, and a confidence score. Each grid 
    // cell also predicts which class each bounding box belongs to.
    //
    // The "features" array therefore contains (numClasses + 5)*boxesPerCell 
    // values for each grid cell, i.e. 125 (425) channels. The total features array
    // contains 13x13x125 (19x19x425) elements (actually x128 (x428) instead of x125 (x425) because in
    // Metal the number of channels must be a multiple of 4).
    
    let blockSize = Float(self.blockSize)
    
    for cy in 0..<gridHeight {
      for cx in 0..<gridWidth {
        for b in 0..<boxesPerCell {
          
          // The 13x13x125 image is arranged in planes of 4 channels. First are
          // channels 0-3 for the entire image, then channels 4-7 for the whole
          // image, then channels 8-11, and so on. Since we have 128 (428) channels,
          // there are 128/4 (428/4) = 32 (107) of these planes (a.k.a. texture slices).
          //
          //    0123 0123 0123 ... 0123    ^
          //    0123 0123 0123 ... 0123    |
          //    0123 0123 0123 ... 0123    13 (19) rows
          //    ...                        |
          //    0123 0123 0123 ... 0123    v
          //    4567 4557 4567 ... 4567
          //    etc
          //    <----- 13 (19) columns ---->
          //
          // For the first bounding box (b=0) we have to read channels 0-24, 
          // for b=1 we have to read channels 25-49, and so on. Unfortunately,
          // these 25 channels are spread out over multiple slices. We use a
          // helper function to find the correct place in the features array.
          // (Note: It might be quicker / more convenient to transpose this
          // array so that all 125 channels are stored consecutively instead
          // of being scattered over multiple texture slices.)
          let channel = b*(numClasses + 5)
          let tx = features[offset(channel, cx, cy)]
          let ty = features[offset(channel + 1, cx, cy)]
          let tw = features[offset(channel + 2, cx, cy)]
          let th = features[offset(channel + 3, cx, cy)]
          let tc = features[offset(channel + 4, cx, cy)]
          
          // The predicted tx and ty coordinates are relative to the location 
          // of the grid cell; we use the logistic sigmoid to constrain these 
          // coordinates to the range 0 - 1. Then we add the cell coordinates 
          // (0-12) and multiply by the number of pixels per grid cell (32).
          // Now x and y represent center of the bounding box in the original
          // 416x416 image space.
          let x = (Float(cx) + Math.sigmoid(tx)) * blockSize
          let y = (Float(cy) + Math.sigmoid(ty)) * blockSize
          
          // The size of the bounding box, tw and th, is predicted relative to
          // the size of an "anchor" box. Here we also transform the width and
          // height into the original 416x416 image space.
          let w = exp(tw) * anchors[2*b    ] * blockSize
          let h = exp(th) * anchors[2*b + 1] * blockSize
          
          // The confidence value for the bounding box is given by tc. We use
          // the logistic sigmoid to turn this into a percentage.
          let confidence = Math.sigmoid(tc)
          
          // Gather the predicted classes for this anchor box and softmax them,
          // so we can interpret these numbers as percentages.
          var classes = [Float](repeating: 0, count: numClasses)
          for c in 0..<numClasses {
            classes[c] = features[offset(channel + 5 + c, cx, cy)]
          }
          classes = Math.softmax(classes)
          
          // Find the index of the class with the largest score.
          let (detectedClass, bestClassScore) = classes.argmax()
          
          // Combine the confidence score for the bounding box, which tells us
          // how likely it is that there is an object in this box (but not what
          // kind of object it is), with the largest class prediction, which
          // tells us what kind of object it detected (but not where).
          let confidenceInClass = bestClassScore * confidence
          
          // Since we compute 13x13x5 = 845 bounding boxes, we only want to
          // keep the ones whose combined score is over a certain threshold.
          if confidenceInClass > 0.3 {
            let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                              width: CGFloat(w), height: CGFloat(h))
            
            let prediction = Prediction(classIndex: detectedClass,
                                        score: confidenceInClass,
                                        rect: rect)
            predictions.append(prediction)
          }
        }
      }
    }
    
    // We already filtered out any bounding boxes that have very low scores, 
    // but there still may be boxes that overlap too much with others. We'll
    // use "non-maximum suppression" to prune those duplicate bounding boxes.
    var result = NeuralNetworkResult<Prediction>()
    
//    result.debugLayer = featuresImage.toFloatArrayChannels()
//    print(result.debugLayer!.count, "==", 19*19*425)
    
//    let debugImage = model.image(for: debugOut!, inflightIndex: inflightIndex)
//    print("debugImage shape:", debugImage.width, "x", debugImage.height, "x", debugImage.featureChannels)
//    result.debugLayer = debugImage.toFloatArrayChannelsTogether()

    result.predictions = nonMaxSuppression(boxes: predictions, limit: 10, threshold: self.threshold)

    return result
  }

public func fetchResult_new(inflightIndex: Int) -> NeuralNetworkResult<Prediction> {

  let featuresImage = model.outputImage(inflightIndex: inflightIndex)
//  print("featuresImage width", featuresImage.width, "height", featuresImage.height, "channels", featuresImage.featureChannels, "images", featuresImage.numberOfImages)
  let features = featuresImage.toFloatArrayChannelsInterleaved()
  
  #if TINY_YOLO
    assert(features.count == 13*13*128)
  #else
    assert(features.count == 19*19*425)
//    print(features.count, "==", 19*19*425)
  #endif
//  print("features =", features.count, "bytes =", features.count*4)
  // We only run the convolutional part of YOLO on the GPU. The last part of
  // the process is done on the CPU. It should be possible to do this on the
  // GPU too, but it might not be worth the effort.
  
  var predictions = [Prediction]()
  
  #if TINY_YOLO
    let blockSize: Float = 32
    let gridHeight = 13
    let gridWidth = 13
    let boxesPerCell = 5
    let numClasses = 20
  #else
    let blockSize: Float = 32
    let gridHeight = 19
    let gridWidth = 19
    let boxesPerCell = 5
    let numClasses = 80
  #endif
  // This helper function finds the offset in the features array for a given
  // channel for a particular pixel. (See the comment below.)
  func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
    return channel + (x + y * gridWidth) * featuresImage.featureChannels
  }
  
  for cy in 0..<gridHeight {
    for cx in 0..<gridWidth {
      for b in 0..<boxesPerCell {
        
        let channel = b*(numClasses + 5)
        let tx = features[offset(channel, cx, cy)]
        let ty = features[offset(channel + 1, cx, cy)]
        let tw = features[offset(channel + 2, cx, cy)]
        let th = features[offset(channel + 3, cx, cy)]
        let tc = features[offset(channel + 4, cx, cy)]
        
        var box_x = Math.sigmoid(tx)
        var box_y = Math.sigmoid(ty)
        var box_w = exp(tw)
        var box_h = exp(th)
        let box_confidence = Math.sigmoid(tc)
        var box_class_probs = [Float](repeating: 0, count: numClasses)
        for c in 0..<numClasses {
          box_class_probs[c] = features[offset(channel + 5 + c, cx, cy)]
        }
        box_class_probs = Math.softmax(box_class_probs)
        box_x = (box_x + Float(cx)) / Float(gridWidth)
        box_y = (box_y + Float(cy)) / Float(gridHeight)
        box_w = box_w * anchors[2*b] / Float(gridWidth)
        box_h = box_h * anchors[2*b+1] / Float(gridHeight)
        
        let (best_class, best_prob) = box_class_probs.argmax()
        let best_score = best_prob * box_confidence
        
        if best_score > 0.3 {
          let rect = CGRect(x: CGFloat(box_x - box_w/2),
                            y: CGFloat(box_y - box_h/2),
                            width: CGFloat(box_w),
                            height: CGFloat(box_h))
          
          let prediction = Prediction(classIndex: best_class,
                                      score: best_score,
                                      rect: rect)
          
          predictions.append(prediction)
        }
      }
    }
  }
  
  // We already filtered out any bounding boxes that have very low scores,
  // but there still may be boxes that overlap too much with others. We'll
  // use "non-maximum suppression" to prune those duplicate bounding boxes.
  var result = NeuralNetworkResult<Prediction>()
  
//  result.debugLayer = featuresImage.toFloatArrayChannelsTogether()
//  print(result.debugLayer!.count, "==", 19*19*425)
  
//  let debugImage = model.image(for: debugOut!, inflightIndex: inflightIndex)
//  print("debugImage shape:", debugImage.width, "x", debugImage.height, "x", debugImage.featureChannels)
//  result.debugLayer = debugImage.toFloatArrayChannelsTogether()
  
  #if TINY_YOLO
    result.predictions = nonMaxSuppression(boxes: predictions, limit: 10, threshold: 0.5)
  #else
    result.predictions = nonMaxSuppression(boxes: predictions, limit: 10, threshold: 0.5)
  #endif
  return result
}

  
/**
 Removes bounding boxes that overlap too much with other boxes that have
 a higher score.
 
 Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc
 
 - Parameters:
 - boxes: an array of bounding boxes and their scores
 - limit: the maximum number of boxes that will be selected
 - threshold: used to decide whether boxes overlap too much
 */
func nonMaxSuppression(boxes: [YOLO.Prediction], limit: Int, threshold: Float) -> [YOLO.Prediction] {
  
  // Do an argsort on the confidence scores, from high to low.
  let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }
  
  var selected: [YOLO.Prediction] = []
  var active = [Bool](repeating: true, count: boxes.count)
  var numActive = active.count
  
  // The algorithm is simple: Start with the box that has the highest score. 
  // Remove any remaining boxes that overlap it more than the given threshold
  // amount. If there are any boxes left (i.e. these did not overlap with any
  // previous boxes), then repeat this procedure, until no more boxes remain 
  // or the limit has been reached.
  outer: for i in 0..<boxes.count {
    if active[i] {
      let boxA = boxes[sortedIndices[i]]
      selected.append(boxA)
      if selected.count >= limit { break }
      
      for j in i+1..<boxes.count {
        if active[j] {
          let boxB = boxes[sortedIndices[j]]
          if IOU(a: boxA.rect, b: boxB.rect) > threshold {
            active[j] = false
            numActive -= 1
            if numActive <= 0 { break outer }
          }
        }
      }
    }
  }
  return selected
}
}
