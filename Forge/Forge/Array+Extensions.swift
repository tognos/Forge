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
  return [a.count, a[0].count]
}

public func shape<T>(_ a : [[[T]]]) -> [Int] {
  return [a.count, a[0].count, a[0][0].count]
}
public func shape<T>(_ a : [[[[T]]]]) -> [Int] {
  return [a.count, a[0].count, a[0][0].count, a[0][0][0].count]
}

public func quadruple<T>(_ a : [T]) -> (T, T, T, T) {
  precondition(a.count == 4, "quadruple input array must have a length of 4")
  return (a[0], a[1], a[2], a[3])
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

public func makeArray<T>(dim: (Int, Int), value: T) -> [[T]] {
  return makeArray(dim: dim.0, value: makeArray(dim: dim.1, value: value))
}

public func makeArray<T>(dim: (Int, Int, Int), value: T) -> [[[T]]] {
  return makeArray(dim: dim.0, value: makeArray(dim: (dim.1, dim.2), value: value))
}

public func makeArray<T>(dim: (Int, Int, Int, Int), value: T) -> [[[[T]]]] {
  return makeArray(dim: dim.0, value: makeArray(dim: (dim.1, dim.2, dim.3), value: value))
}

public func sliceArray<T>(_ a : [[[[T]]]], from: (Int, Int, Int, Int), size: (Int, Int, Int, Int)) -> [[[[T]]]] {
  let srcDims = shape(a)
  precondition(from.0 >= 0 && from.1 >= 0 && from.2 >= 0 && from.3 >= 0 &&
               from.0 + size.0 <= srcDims[0] && from.1 + size.1 <= srcDims[1] &&
               from.2 + size.2 <= srcDims[2] && from.3 + size.3 <= srcDims[3],
               "from + size must be smaller than dims")
  var result = makeArray(dim: size, value: a[0][0][0][0])
  for l in 0..<size.0 {
    for k in 0..<size.1 {
      for j in 0..<size.2 {
        for i in 0..<size.3 {
          result[l][k][j][i] = a[from.0 + l][from.1 + k][from.2 + j][from.3 + i]
        }
      }
    }
  }
  return result
}

public func transposed<T>(_ a : [[T]]) -> [[T]] {
  let dims = shape(a)
  let srcColumns = dims[1]
  let srcRows = dims[0]
  var result = makeArray(dim: (srcColumns, srcRows), value: a[0][0])
  for j in 0..<srcRows {
    for i in 0..<srcColumns {
      result[i][j] = a[j][i]
    }
  }
  return result
}

public func transposed<T>(_ a : [[[T]]], axes: (Int, Int, Int)) -> [[[T]]] {
  let srcDims = shape(a)
  let (srcPlanes, srcRows, srcColumns) = (srcDims[0],srcDims[1], srcDims[2])
  
  precondition(axes.0 != axes.1 && axes.0 != axes.2 && axes.1 != axes.2, "axes must be different")
  precondition((0...2).contains(axes.0) && (0...2).contains(axes.1) && (0...2).contains(axes.2), "axes must 0, 1 or 2")
  
  let destDims = (srcDims[axes.0], srcDims[axes.1], srcDims[axes.2])
  let axisArray = [axes.0, axes.1, axes.2]

  var result = makeArray(dim: destDims, value: a[0][0][0])
  precondition(shape(result) == [destDims.0, destDims.1, destDims.2])
  for k in 0..<srcPlanes {
    for j in 0..<srcRows {
      for i in 0..<srcColumns {
        let src = [k, j, i]
        let (dk, dj, di) = (src[axisArray[0]], src[axisArray[1]], src[axisArray[2]])
        result[dk][dj][di] = a[k][j][i]
      }
    }
  }
  return result
}

public func transposed<T>(_ a : [[[[T]]]], axes: (Int, Int, Int, Int)) -> [[[[T]]]] {
  let srcDims = shape(a)
  let (srcCubes, srcPlanes, srcRows, srcColumns) = (srcDims[0],srcDims[1], srcDims[2], srcDims[3])
  
  precondition(axes.0 != axes.1 && axes.0 != axes.2 && axes.0 != axes.3 &&
                         axes.1 != axes.2 && axes.1 != axes.3 &&
                         axes.2 != axes.3, "all axes must be different")
  precondition((0...3).contains(axes.0) && (0...3).contains(axes.1) &&
               (0...3).contains(axes.2) && (0...3).contains(axes.3), "axes must 0, 1, 2 or 3")
  
  let destDims = (srcDims[axes.0], srcDims[axes.1], srcDims[axes.2], srcDims[axes.3])
  let axisArray = [axes.0, axes.1, axes.2, axes.3]
  
  var result = makeArray(dim: destDims, value: a[0][0][0][0])
  
  precondition(shape(result) == [destDims.0, destDims.1, destDims.2, destDims.3])

  for l in 0..<srcCubes {
    for k in 0..<srcPlanes {
      for j in 0..<srcRows {
        for i in 0..<srcColumns {
          let src = [l, k, j, i]
          let (dl, dk, dj, di) = (src[axisArray[0]], src[axisArray[1]], src[axisArray[2]], src[axisArray[3]])
          result[dl][dk][dj][di] = a[l][k][j][i]
        }
      }
    }
  }
  return result
}

public func reverse(axes: (Int, Int, Int, Int)) -> (Int, Int ,Int, Int) {
  let axisArray = [axes.0, axes.1, axes.2, axes.3]
  let revAxes = axisArray.argsort(by: <)
  return (revAxes[0], revAxes[1], revAxes[2], revAxes[3])
}

extension Array {

  // reshape an array into an Array of Arrays
  // The new array will contain dim1 arrays of size dim0
  // Array.count/dim1 must be equal to dim1 when dim0 is != -1
  // When dim0 == -1 it is computed from dim1 and the array size
  public func reshaped(_ dim0_parm : Int,_ dim1 : Int) -> [[Element]] {
    precondition(self.count % dim1 == 0, "array size must be divisible by dim0")
    var dim0 = dim0_parm
    if (dim0 != -1) {
      precondition(self.count / dim1 == dim0, "array size / dim 1 must equal dim0")
    } else {
      dim0 = self.count / dim1
    }

    let newArraySize = self.count / dim1
    var result : [[Element]] = []
    for i in 0..<newArraySize {
      let new_slice = self[(i * dim1)..<((i + 1)*dim1)]
      result.append(Array(new_slice))
    }
    precondition(result.count == dim0, "internal error")
    return result
  }
  public func reshaped(_ dims :(Int, Int)) -> [[Element]] {
    return reshaped(dims.0, dims.1)
  }
  public func reshaped(_ dims :(Int, Int, Int)) -> [[[Element]]] {
    return reshaped(dims.0, dims.1, dims.2)
  }
  public func reshaped(_ dim0: Int, _ dim1: Int, _ dim2: Int) -> [[[Element]]] {
    return reshaped(-1, dim2).reshaped(dim0, dim1)
  }
  public func reshaped(_ dims :(Int, Int, Int, Int)) -> [[[[Element]]]] {
    return reshaped(dims.0, dims.1, dims.2, dims.3)
  }
  public func reshaped(_ dim0: Int, _ dim1: Int, _ dim2: Int, _ dim3: Int) -> [[[[Element]]]] {
    return reshaped(-1, dim3).reshaped(-1, dim2).reshaped(dim0, dim1)
  }
}



