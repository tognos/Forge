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
    let expected34 : [[Float]] = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
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
    if !(reshaped234 == expected234) {
      fatalError("Assertion failed: \(reshaped234) not equal to \(expected234)")
    }
    
    let expected2341 : [[[[Float]]]] = [[[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]],[[19,20],[21,22],[23,24]]]]
    let reshaped2341 = b.reshaped(2,3,4,1)
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
    
  }
}
