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
  func testMakeArray() {
    print("\(self).\(#function)")
    precondition(shape(makeArray(dim: 5, value: 0)) == [5])
    precondition(shape(makeArray(dim: (2, 3), value: 0)) == [2,3])
    //print(shape(makeArray3D(dim: (2, 3, 4), value: 0)))
    precondition(shape(makeArray(dim: (2, 3, 4), value: 0)) == [2,3,4])
    precondition(shape(makeArray(dim: (2, 3, 4, 5), value: 0)) == [2,3,4,5])
  }
    
  func testReshape() {
    print("\(self).\(#function)")
    let a : [Float] = [1,2,3,4,5,6,7,8,9,10,11,12]
    let expected34 : [[Float]] = [[1,2,3],
                                  [4,5,6],
                                  [7,8,9],
                                  [10,11,12]]
    if shape(expected34) != [4,3] {
      fatalError("Assertion failed: shape mismatch")
    }
    let reshaped34 = a.reshaped(4,3)
    if !(reshaped34 == expected34) {
      fatalError("Assertion failed: \(reshaped34) not equal to \(expected34)")
    }
    
    let b : [Float] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
    let expected2_12 : [[Float]] = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12],[13,14],[15,16],[17,18],[19,20],[21,22],[23,24]]
    let reshaped2_12 = b.reshaped(-1,2)
    if !(reshaped2_12 == expected2_12) {
      fatalError("Assertion failed: \(reshaped2_12) not equal to \(expected2_12)")
    }
    
    let expected234 : [[[Float]]] = [[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]],[[19,20],[21,22],[23,24]]]
    let reshaped234 = b.reshaped(4,3,2)
    //print("reshaped234=\(reshaped234)")
    //print("shape(reshaped234)=\(shape(reshaped234))")
    if shape(reshaped234) != [4,3,2] {
      fatalError("Assertion failed: shape mismatch")
    }
    if !(reshaped234 == expected234) {
      fatalError("Assertion failed: \(reshaped234) not equal to \(expected234)")
    }
    
    let expected2341 : [[[[Float]]]] = [[[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]],[[19,20],[21,22],[23,24]]]]
    let reshaped2341 = b.reshaped(1,4,3,2)
    print("reshaped2341=\(reshaped2341)")
    print("shape(reshaped2341)=\(shape(reshaped2341))")
    if shape(reshaped2341) != [1,4,3,2] {
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
    let reshaped2342 = c.reshaped(2,4,3,2)
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
      let trans_out_43 = transposed(trans_in_34)
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
      precondition(shape(trans_in_234) == [4,3,2])
      do {
        // 3D-identity transpose
        let trans_out_234 = transposed(trans_in_234, axes: (0,1,2))
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
        
        let trans_out_432 = transposed(trans_in_234, axes: (2,1,0))
        if !(trans_out_432 == trans_out_expected_432) {
          preconditionFailure("Assertion failed: \(trans_out_432) not equal to \(trans_out_expected_432)")
        }
      }
      do {
        // 3D-all dims transpose
        
        let trans_out_expected_423: [[[Float]]] =  [[[1, 7, 13, 19],
                                                     [2, 8, 14, 20]],
                                                    [[3, 9, 15, 21],
                                                     [4, 10, 16, 22]],
                                                    [[5, 11, 17, 23],
                                                     [6, 12, 18, 24]]]
        
        let trans_out_423 = transposed(trans_in_234, axes: (1,2,0))
        if !(trans_out_423 == trans_out_expected_423) {
          preconditionFailure("Assertion failed: \(trans_out_423) not equal to \(trans_out_expected_423)")
        }
        let back_again = transposed(transposed(trans_out_423, axes: (1,2,0)), axes: (1,2,0))
        if !(back_again == trans_in_234) {
          preconditionFailure("Assertion failed: \(back_again) not equal to \(trans_in_234)")
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
        // 4D-identity transpose
        let trans_out_3252 = transposed(trans_in_3252, axes: (0,1,2,3))
        if !(trans_out_3252 == trans_in_3252) {
          fatalError("Assertion failed: \(trans_out_3252) not equal to \(trans_in_3252)")
        }
      }
      do {
        // 4D-reverse dims transpose
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
        
        let trans_out_2523 = transposed(trans_in_3252, axes: (3,2,1,0))
        if !(trans_out_2523 == trans_out_expected_2523) {
          preconditionFailure("Assertion failed: \(trans_out_2523) not equal to \(trans_out_expected_2523)")
        }
      }
      do {
        // 4D-all dims transpose
        
        let trans_out_expected_2523 : [[[[Float]]]] = [[[[111, 121],
                                                         [112, 122],
                                                         [113, 123]],
                                                        [[114, 124],
                                                         [115, 125],
                                                         [116, 126]]],
                                                       [[[211, 221],
                                                         [212, 222],
                                                         [213, 223]],
                                                        [[214, 224],
                                                         [215, 225],
                                                         [216, 226]]],
                                                       [[[311, 321],
                                                         [312, 322],
                                                         [313, 323]],
                                                        [[314, 324],
                                                         [315, 325],
                                                         [316, 326]]],
                                                       [[[411, 421],
                                                         [412, 422],
                                                         [413, 423]],
                                                        [[414, 424],
                                                         [415, 425],
                                                         [416, 426]]],
                                                       [[[511, 521],
                                                         [512, 522],
                                                         [513, 523]],
                                                        [[514, 524],
                                                         [515, 525],
                                                         [516, 526]]]]
        
        let trans_out_2523 = transposed(trans_in_3252, axes: (1,2,3,0))
        if !(trans_out_2523 == trans_out_expected_2523) {
          preconditionFailure("Assertion failed: \(trans_out_2523) not equal to \(trans_out_expected_2523)")
        }
        var multi_trans = trans_in_3252
        for _ in 1...4 {
          multi_trans = transposed(multi_trans, axes: (1,2,3,0))
        }
        if !(multi_trans == trans_in_3252) {
          preconditionFailure("Assertion failed: \(multi_trans) not equal to \(trans_in_3252)")
        }
      }}

  }
  func testSlice() {
    print("\(self).\(#function)")
    do {
      // 2D-transpose
      
      let slice_in_3252 : [[[[Int]]]] = [[[[111,112,113],
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
      let inShape = shape(slice_in_3252)
      //print("shape of slice test input:",inShape)
      let slice_out = sliceArray( slice_in_3252, from: (0,0,0,0), size: quadruple(inShape) )
      if !(slice_out == slice_in_3252) {
        fatalError("Assertion failed: \(slice_out) not equal to \(slice_in_3252)")
      }
      let slice_out2 = sliceArray( slice_in_3252, from: (0,0,0,0), size: (1, inShape[1], inShape[2], inShape[3]) )
      //print("slice_out2", slice_out2)
      //print("slice_out2 shape", shape(slice_out2))
      let slice_out2_expected = [[[[111, 112, 113],
                                   [114, 115, 116]],
                                  [[211, 212, 213],
                                   [214, 215, 216]],
                                  [[311, 312, 313],
                                   [314, 315, 316]],
                                  [[411, 412, 413],
                                   [414, 415, 416]],
                                  [[511, 512, 513],
                                   [514, 515, 516]]]]
      if !(slice_out2 == slice_out2_expected) {
        fatalError("Assertion failed: \(slice_out2) not equal to \(slice_out2_expected)")
      }
      
      let slice_out3 = sliceArray( slice_in_3252, from: (0,0,0,0), size: (inShape[0], 1, inShape[2], inShape[3]) )
      //print("slice_out3", slice_out3)
      //print("slice_out3 shape", shape(slice_out3))
      let slice_out3_expected = [[[[111, 112, 113], [114, 115, 116]]], [[[121, 122, 123], [124, 125, 126]]]]
      if !(slice_out3 == slice_out3_expected) {
        fatalError("Assertion failed: \(slice_out3) not equal to \(slice_out3_expected)")
      }
      
      let slice_out4 = sliceArray( slice_in_3252, from: (0,0,0,0), size: (inShape[0], inShape[1], 1, inShape[3]) )
      //print("slice_out4", slice_out4)
      //print("slice_out4 shape", shape(slice_out4))
      let slice_out4_expected = [[[[111, 112, 113]],
                                  [[211, 212, 213]],
                                  [[311, 312, 313]],
                                  [[411, 412, 413]],
                                  [[511, 512, 513]]],
                                 [[[121, 122, 123]],
                                  [[221, 222, 223]],
                                  [[321, 322, 323]],
                                  [[421, 422, 423]],
                                  [[521, 522, 523]]]]
      if !(slice_out4 == slice_out4_expected) {
        fatalError("Assertion failed: \(slice_out4) not equal to \(slice_out4_expected)")
      }
      
      let slice_out5 = sliceArray( slice_in_3252, from: (0,0,0,0), size: (inShape[0], inShape[1], inShape[2], 1) )
      //print("slice_out5", slice_out5)
      //print("slice_out5 shape", shape(slice_out5))
      let slice_out5_expected = [[[[111], [114]], [[211], [214]], [[311], [314]], [[411], [414]], [[511], [514]]],
                                 [[[121], [124]], [[221], [224]], [[321], [324]], [[421], [424]], [[521], [524]]]]
      if !(slice_out5 == slice_out5_expected) {
        fatalError("Assertion failed: \(slice_out5) not equal to \(slice_out5_expected)")
      }
      
      let slice_out6 = sliceArray( slice_in_3252, from: (1,1,1,1), size: (inShape[0]-1, inShape[1]-1, inShape[2]-1, inShape[3]-1) )
      //print("slice_out6", slice_out6)
      //print("slice_out6 shape", shape(slice_out6))
      let slice_out6_expected = [[[[225, 226]], [[325, 326]], [[425, 426]], [[525, 526]]]]
      if !(slice_out6 == slice_out6_expected) {
        fatalError("Assertion failed: \(slice_out6) not equal to \(slice_out6_expected)")
      }
    }
  }
}
