import Foundation
import Forge

class ArrayTests {
  func testArgmax() {
    print("\(self).\(#function)")

    let a: [Float] = [ 2, 0, 7, -1, 8, 3, 7, 5 ]

    let (maxIndex, maxValue) = a.argmax()
    assertEqual(maxIndex, 4)
    assertEqual(maxValue, 8)
  }

  func testArgsort() {
    print("\(self).\(#function)")

    let a = [ 2, 0, 7, -1, 8, 3, -2, 5 ]
    let s = a.argsort(by: <)

    let i = [ 6, 3, 1, 0, 5, 7, 2, 4 ]   // these are indices!
    assertEqual(s, i)
  }

  func testGather() {
    print("\(self).\(#function)")

    let a = [ 2, 0, 7, -1, 8, 3, -2, 5 ]
    let i = [ 6, 3, 1, 0, 5, 7, 2, 4 ]   // these are indices!
    let g = a.gather(indices: i)

    let e = [ -2, -1, 0, 2, 3, 5, 7, 8 ]
    assertEqual(g, e)
  }
    
  func testReshape() {
    print("\(self).\(#function)")
    let a : [Float] = [1,2,3,4,5,6,7,8,9,10,11,12]
    let expected34 : [[Float]] = [[1,2,3],
                                  [4,5,6],
                                  [7,8,9],
                                  [10,11,12]]
    if shape(expected34) != [3,4] {
      fatalError("Assertion failed: shape mismatch")
    }
    let reshaped34 = a.reshaped(3,4)
    if !(reshaped34 == expected34) {
      fatalError("Assertion failed: \(reshaped34) not equal to \(expected34)")
    }
    
    let b : [Float] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    let expected2_12 : [[Float]] = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,22],[23,24]]
    let reshaped2_12 = b.reshaped(2,-1)
    if !(reshaped2_12 == expected2_12) {
      fatalError("Assertion failed: \(reshaped2_12) not equal to \(expected2_12)")
    }
    
    let expected234 : [[[Float]]] = [[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]],[[19,20],[21,22],[23,24]]]
    let reshaped234 = b.reshaped(2,3,4)
    if shape(reshaped234) != [2,3,4] {
      fatalError("Assertion failed: shape mismatch")
    }
    if !(reshaped234 == expected234) {
      fatalError("Assertion failed: \(reshaped234) not equal to \(expected234)")
    }
    
    let expected2341 : [[[[Float]]]] = [[[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]],[[19,20],[21,22],[23,24]]]]
    let reshaped2341 = b.reshaped(2,3,4,1)
    if shape(reshaped2341) != [2,3,4,1] {
      fatalError("Assertion failed: shape mismatch")
    }
    if !(reshaped2341 == expected2341) {
      fatalError("Assertion failed: \(reshaped2341) not equal to \(expected2341)")
    }
    
    let c : [Float] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,
                       201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224]
    let expected2342 : [[[[Float]]]] = [[[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],
                                         [[13,14],[15,16],[17,18]],[[19,20],[21,22],[23,24]]],
                                        [[[201,202],[203,204],[205,206]],[[207,208],[209,210],[211,212]],
                                         [[213,214],[215,216],[217,218]],[[219,220],[221,222],[223,224]]]]
    let reshaped2342 = c.reshaped(2,3,4,2)
    if !(reshaped2342 == expected2342) {
      fatalError("Assertion failed: \(reshaped2342) not equal to \(expected2342)")
    }
    let flat : [Float] = flattened(reshaped2342)
    if !(flat == c) {
      fatalError("Assertion failed: \(flat) not equal to \(c)")
    }
  }
  func testTranspose() {
    print("\(self).\(#function)")
    do {
      // 2D-transpose

      let trans_in_34  : [[Float]] = [[1,2,3],
                                      [4,5,6],
                                      [7,8,9],
                                      [10,11,12]]
      let trans_out_expected_43  : [[Float]] = [[1, 4, 7, 10],
                                                [2, 5, 8, 11],
                                                [3, 6, 9, 12]]
      let trans_out_43 = transposed2D(trans_in_34)
      if !(trans_out_43 == trans_out_expected_43) {
        fatalError("Assertion failed: \(trans_out_43) not equal to \(trans_out_expected_43)")
      }
    }
    do {
      // 3D-Transpose
      let trans_in_234 : [[[Float]]] = [[[1,2],
                                         [3,4],
                                         [5,6]],
                                        [[7,8],
                                         [9,10],
                                         [11,12]],
                                        [[13,14],
                                         [15,16],
                                         [17,18]],
                                        [[19,20],
                                         [21,22],
                                         [23,24]]]
      do {
        // 3D-identity transpose
        let trans_out_234 = transposed3D(trans_in_234, axes: (0,1,2))
        if !(trans_out_234 == trans_in_234) {
          fatalError("Assertion failed: \(trans_out_234) not equal to \(trans_in_234)")
        }
      }
      do {
        // 3D-reverse dims transpose
        let trans_out_expected_432 : [[[Float]]] = [[[1, 7, 13, 19],
                                                     [3, 9, 15, 21],
                                                     [5, 11, 17, 23]],
                                                    [[2, 8, 14, 20],
                                                     [4, 10, 16, 22],
                                                     [6, 12, 18, 24]]]
        
        let trans_out_432 = transposed3D(trans_in_234, axes: (2,1,0))
        if !(trans_out_432 == trans_out_expected_432) {
          preconditionFailure("Assertion failed: \(trans_out_432) not equal to \(trans_out_expected_432)")
        }
      }
      do {
        // 3D-all dims transpose
        
        let trans_out_expected_342: [[[Float]]] = [[[1, 3, 5],
                                                    [7, 9, 11],
                                                    [13, 15, 17],
                                                    [19, 21, 23]],
                                                   [[2, 4, 6],
                                                    [8, 10, 12],
                                                    [14, 16, 18],
                                                    [20, 22, 24]]]
        
        let trans_out_342 = transposed3D(trans_in_234, axes: (1,2,0))
        if !(trans_out_342 == trans_out_expected_342) {
          preconditionFailure("Assertion failed: \(trans_out_342) not equal to \(trans_out_expected_342)")
        }
      }
    }
    do {
      let trans_in_3252 : [[[[Float]]]] = [[[[111,112,113],
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
      do {
        // 3D-identity transpose
        let trans_out_3252 = transposed4D(trans_in_3252, axes: (0,1,2,3))
        if !(trans_out_3252 == trans_in_3252) {
          fatalError("Assertion failed: \(trans_out_3252) not equal to \(trans_in_3252)")
        }
      }
      do {
        // 3D-reverse dims transpose
        let trans_out_expected_2523 : [[[[Float]]]] = [[[[111, 121],
                                                         [211, 221],
                                                         [311, 321],
                                                         [411, 421],
                                                         [511, 521]],
                                                        [[114, 124],
                                                         [214, 224],
                                                         [314, 324],
                                                         [414, 424],
                                                         [514, 524]]],
                                                       [[[112, 122],
                                                         [212, 222],
                                                         [312, 322],
                                                         [412, 422],
                                                         [512, 522]],
                                                        [[115, 125],
                                                         [215, 225],
                                                         [315, 325],
                                                         [415, 425],
                                                         [515, 525]]],
                                                       [[[113, 123],
                                                         [213, 223],
                                                         [313, 323],
                                                         [413, 423],
                                                         [513, 523]],
                                                        [[116, 126],
                                                         [216, 226],
                                                         [316, 326],
                                                         [416, 426],
                                                         [516, 526]]]]
        
        let trans_out_2523 = transposed4D(trans_in_3252, axes: (3,2,1,0))
        if !(trans_out_2523 == trans_out_expected_2523) {
          preconditionFailure("Assertion failed: \(trans_out_2523) not equal to \(trans_out_expected_2523)")
        }
      }
      do {
        // 3D-all dims transpose
        
        let trans_out_expected_2523 : [[[[Float]]]] = [[[[111, 114],
                                                         [211, 214],
                                                         [311, 314],
                                                         [411, 414],
                                                         [511, 514]],
                                                        [[121, 124],
                                                         [221, 224],
                                                         [321, 324],
                                                         [421, 424],
                                                         [521, 524]]],
                                                       [[[112, 115],
                                                         [212, 215],
                                                         [312, 315],
                                                         [412, 415],
                                                         [512, 515]],
                                                        [[122, 125],
                                                         [222, 225],
                                                         [322, 325],
                                                         [422, 425],
                                                         [522, 525]]],
                                                       [[[113, 116],
                                                         [213, 216],
                                                         [313, 316],
                                                         [413, 416],
                                                         [513, 516]],
                                                        [[123, 126],
                                                         [223, 226],
                                                         [323, 326],
                                                         [423, 426],
                                                         [523, 526]]]]
        
        let trans_out_2523 = transposed4D(trans_in_3252, axes: (1,2,3,0))
        if !(trans_out_2523 == trans_out_expected_2523) {
          preconditionFailure("Assertion failed: \(trans_out_2523) not equal to \(trans_out_expected_2523)")
        }
      }}

  }
}
