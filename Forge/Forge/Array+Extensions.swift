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

// Array.joined() does not work on 4-dimensional arrays, so we help it out a bit

public func shape<T>(_ a : [T]) -> [Int] {
  return [a.count]
  
}
public func shape<T>(_ a : [[T]]) -> [Int] {
  return [a[0].count, a.count]
}

public func shape<T>(_ a : [[[T]]]) -> [Int] {
  return [a[0][0].count, a[0].count, a.count]
}
public func shape<T>(_ a : [[[[T]]]]) -> [Int] {
  return [a[0][0][0].count, a[0][0].count, a[0].count, a.count]
}

public func flattened<T>(_ a : [[T]]) -> [T] {
  return Array<T>(a.joined())
}

public func flattened<T>(_ a : [[[T]]]) -> [T] {
  return Array<T>(a.joined().joined())
}

public func flattened<T>(_ a : [[[[T]]]]) -> [T] {
  return Array<T>(a.joined().joined().joined())
}

public func makeArray<T>(dim: Int, value: T) -> [T] {
  return [T](repeating: value, count: dim)
}

public func makeArray2D<T>(dim: (Int, Int), value: T) -> [[T]] {
  return makeArray(dim: dim.1, value: makeArray(dim: dim.0, value: value))
}

public func makeArray3D<T>(dim: (Int, Int, Int), value: T) -> [[[T]]] {
  return makeArray(dim: dim.2, value: makeArray2D(dim: (dim.0, dim.1), value: value))
}

public func makeArray4D<T>(dim: (Int, Int, Int, Int), value: T) -> [[[[T]]]] {
  return makeArray(dim: dim.3, value: makeArray3D(dim: (dim.0, dim.1, dim.2), value: value))
}

public func transposed2D<T>(_ a : [[T]]) -> [[T]] {
  var result : [[T]] = [[]]
  let dims = shape(a)
  let srcColumns = dims[0]
  let srcRows = dims[1]
  result = makeArray2D(dim: (srcRows, srcColumns), value: a[0][0])
  for j in 0..<srcRows {
    for i in 0..<srcColumns {
      result[i][j] = a[j][i]
    }
  }
  return result
}

public func transposed3D<T>(_ a : [[[T]]], axes: (Int, Int, Int)) -> [[[T]]] {
  
  var result : [[[T]]] = [[[]]]
  let srcDims = shape(a)
  let (srcColumns, srcRows, srcPlanes) = (srcDims[0],srcDims[1], srcDims[2])
  
  precondition(axes.0 != axes.1 && axes.0 != axes.2 && axes.1 != axes.2, "axes must be different")
  precondition((0...2).contains(axes.0) && (0...2).contains(axes.1) && (0...2).contains(axes.2), "axes must 0, 1 or 2")
  
  let destDims = (srcDims[axes.0], srcDims[axes.1], srcDims[axes.2])
  let axisArray = [axes.0, axes.1, axes.2]
  let destAxes = axisArray.argsort(by: <)
  
  result = makeArray3D(dim: destDims, value: a[0][0][0])
  precondition(shape(result) == [destDims.0, destDims.1, destDims.2])
  for k in 0..<srcPlanes {
    for j in 0..<srcRows {
      for i in 0..<srcColumns {
        let src = [k, j, i]
        let (di, dj, dk) = (src[destAxes[0]], src[destAxes[1]], src[destAxes[2]])
        result[di][dj][dk] = a[k][j][i]
      }
    }
  }
  return result
}

public func transposed4D<T>(_ a : [[[[T]]]], axes: (Int, Int, Int, Int)) -> [[[[T]]]] {
  
  var result : [[[[T]]]] = [[[[]]]]
  let srcDims = shape(a)
  let (srcColumns, srcRows, srcPlanes, srcCubes) = (srcDims[0],srcDims[1], srcDims[2], srcDims[3])
  
  precondition(axes.0 != axes.1 && axes.0 != axes.2 && axes.0 != axes.3 &&
                         axes.1 != axes.2 && axes.1 != axes.3 &&
                         axes.2 != axes.3, "all axes must be different")
  precondition((0...3).contains(axes.0) && (0...3).contains(axes.1) &&
               (0...3).contains(axes.2) && (0...3).contains(axes.3), "axes must 0, 1, 2 or 3")
  
  let destDims = (srcDims[axes.0], srcDims[axes.1], srcDims[axes.2], srcDims[axes.3])
  let axisArray = [axes.0, axes.1, axes.2, axes.3]
  let destAxes = axisArray.argsort(by: <)
  
  result = makeArray4D(dim: destDims, value: a[0][0][0][0])
  precondition(shape(result) == [destDims.0, destDims.1, destDims.2, destDims.3])
  for l in 0..<srcCubes {
    for k in 0..<srcPlanes {
      for j in 0..<srcRows {
        for i in 0..<srcColumns {
          let src = [l, k, j, i]
          let (di, dj, dk, dl) = (src[destAxes[0]], src[destAxes[1]], src[destAxes[2]], src[destAxes[3]])
          result[di][dj][dk][dl] = a[l][k][j][i]
        }
      }
    }
  }
  return result
}


extension Array {

  // reshape an array into an Array of Arrays
  // The new array will contain dim1 arrays of size dim0
  // Array.count/dim0 must be equal to dim1 when dim1 is != -1
  // When dim1 == -1 it is computed from dim0 and the array size
  public func reshaped(_ dim0 : Int,_ dim1_parm : Int) -> [[Element]] {
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
    return result
  }
  public func reshaped(_ dim0: Int, _ dim1: Int, _ dim2: Int) -> [[[Element]]] {
    return reshaped(dim0, -1).reshaped(dim1, dim2)
  }
  public func reshaped(_ dim0: Int, _ dim1: Int, _ dim2: Int, _ dim3: Int) -> [[[[Element]]]] {
    return reshaped(dim0, -1).reshaped(dim1, -1).reshaped(dim2, dim3)
  }
}

