/*
  Copyright (c) 2016-2017 M.I. Hollemans

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to
  deal in the Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
  sell copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
  IN THE SOFTWARE.
*/

import Foundation

extension Array where Element: Comparable {
  /**
    Returns a new array with (index, element) tuples for the `k` elements
    with the highest values.
    
    Useful for getting the top-5 predictions, for example.
    
    You can map the array to labels by writing something like:
    
        array.top(k: 5).map { x -> (String, Float) in (labels[x.0], x.1) }
  */
  public func top(k: Int) -> [(Int, Element)] {
    return Array<(Int, Element)>(
              zip(0..<self.count, self)
             .sorted(by: { a, b -> Bool in a.1 > b.1 })
             .prefix(through: Swift.min(k, self.count) - 1))
  }

  /**
    Returns the index and value of the largest element in the array.
  */
  public func argmax() -> (Int, Element) {
    precondition(self.count > 0)
    var maxIndex = 0
    var maxValue = self[0]
    for i in 1..<self.count {
      if self[i] > maxValue {
        maxValue = self[i]
        maxIndex = i
      }
    }
    return (maxIndex, maxValue)
  }

  /**
    Returns the indices of the array's elements in sorted order.
  */
  public func argsort(by areInIncreasingOrder: (Element, Element) -> Bool) -> [Array.Index] {
    return self.indices.sorted { areInIncreasingOrder(self[$0], self[$1]) }
  }

  /**
    Returns a new array containing the elements at the specified indices.
  */
  public func gather(indices: [Array.Index]) -> [Element] {
    var a = [Element]()
    for i in indices { a.append(self[i]) }
    return a
  }
}


// Some functions to compare up to 4-dimensional arrays
// unfortunately there is not yet really a better way to
// do that in Swift4; maybe Swift5 will bring us a way
// to specialize on associated types in protocols

public func areAlmostEqual<T:FloatingPoint>(_ a1: T, _ a2: T, maxError: T) -> Bool {
  return abs(a1 - a2) <= maxError
}

public func areAlmostEqual<T:FloatingPoint>(_ a1: [T], _ a2: [T], maxError: T, reportUnequal: (String) -> ()) -> Bool {
  if a1.count != a2.count {
    reportUnequal("count mismatch dim 0")
    return false
  }
  for i in 0..<a1.count {
    if !areAlmostEqual(a1[i], a2[i], maxError: maxError) {
      //print(a1[i], a2[i])
      reportUnequal("error > maxError, dim 0 index "+String(i)+", error="+String(describing: a1[i] - a2[i]))
      return false
    }
  }
  return true
}

public func areAlmostEqual<T:FloatingPoint>(_ a1: [[T]], _ a2: [[T]], maxError: T, reportUnequal: (String) -> ()) -> Bool {
  if a1.count != a2.count {
    reportUnequal("count mismatch dim 1")
    return false
  }
  for i in 0..<a1.count {
    if !areAlmostEqual(a1[i], a2[i], maxError: maxError, reportUnequal: reportUnequal) {
      reportUnequal("error > maxError, dim 1 index "+String(i))
      return false
    }
  }
  return true
}

public func areAlmostEqual<T:FloatingPoint>(_ a1: [[[T]]], _ a2: [[[T]]], maxError: T, reportUnequal: (String) -> ()) -> Bool {
  if a1.count != a2.count {
    reportUnequal("count mismatch dim 2")
    return false
  }
  for i in 0..<a1.count {
    if !areAlmostEqual(a1[i], a2[i], maxError: maxError, reportUnequal: reportUnequal) {
      reportUnequal("error > maxError, dim 2 index "+String(i))
      return false
    }
  }
  return true
}

public func areAlmostEqual<T:FloatingPoint>(_ a1: [[[[T]]]], _ a2: [[[[T]]]], maxError: T, reportUnequal: (String) -> ()) -> Bool {
  if a1.count != a2.count {
    reportUnequal("count mismatch dim 3")
    return false
  }
  for i in 0..<a1.count {
    if !areAlmostEqual(a1[i], a2[i], maxError: maxError, reportUnequal: reportUnequal) {
      reportUnequal("val error > maxError, dim 3 index"+String(i))
      return false
    }
  }
  return true
}

public func == <T:Equatable>(_ a1: [[T]], _ a2: [[T]]) -> Bool {
  if a1.count != a2.count {
    print("count mismatch dim 2")
    return false
  }
  for i in 0..<a1.count {
    if !(a1[i] == a2[i]) {
      print("val mismatch dim 2 index",i)
      return false
    }
  }
  return true
}

public func == <T:Equatable>(_ a1: [[[T]]], _ a2: [[[T]]]) -> Bool {
  if a1.count != a2.count {
    print("count mismatch dim 3")
    return false
  }
  for i in 0..<a1.count {
    if !(a1[i] == a2[i]) {
      print("val mismatch dim 3 index",i)
      return false
    }
  }
  return true
}

public func == <T:Equatable>(_ a1: [[[[T]]]], _ a2: [[[[T]]]]) -> Bool {
  if a1.count != a2.count {
    print("count mismatch dim 4")
    return false
  }
  for i in 0..<a1.count {
    if !(a1[i] == a2[i]) {
      print("val mismatch dim 4 index",i)
      return false
    }
  }
  return true
}

extension Array {

  // reshape an array into an Array of Arrays
  // The new array will contain dim1 arrays of size dim0
  // Array.count/dim0 must be equal to dim1 when dim1 is != -1
  // When dim1 == -1 it is computed from dim0 and the array size
  public func reshaped(_ dim0 : Int,_ dim1_parm : Int) -> [[Element]] {
    //print("Reshaping array of size \(count) to \(dim0),\(dim1_parm)")
    precondition(self.count % dim0 == 0, "array size must be divisible by dim0")
    var dim1 = dim1_parm
    if (dim1 != -1) {
      precondition(self.count / dim0 == dim1, "array size / dim 0 must equal dim1")
    } else {
      dim1 = self.count / dim0
    }

    let newArraySize = self.count / dim0
    var result : [[Element]] = []
    for i in 0..<newArraySize {
      let new_slice = self[(i * dim0)..<((i + 1)*dim0)]
      result.append(Array(new_slice))
    }
    precondition(result.count == dim1, "internal error")
//    print(result)
    return result
  }
  public func reshaped(_ dim0: Int, _ dim1: Int, _ dim2: Int) -> [[[Element]]] {
    return reshaped(dim0, -1).reshaped(dim1, dim2)
  }
  public func reshaped(_ dim0: Int, _ dim1: Int, _ dim2: Int, _ dim3: Int) -> [[[[Element]]]] {
    return reshaped(dim0, -1).reshaped(dim1, -1).reshaped(dim2, dim3)
  }
}

