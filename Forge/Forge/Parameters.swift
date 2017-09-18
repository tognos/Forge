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

public enum ParameterType {
  case weights
  case biases
}

public protocol ParameterData {
  var pointer: UnsafeMutablePointer<Float> { get }
}

public class ParameterLoaderBundle: ParameterData {
  private var fileSize: Int
  private var fd: CInt!
  private var hdr: UnsafeMutableRawPointer!
  private(set) public var pointer: UnsafeMutablePointer<Float>

  /**
    Load layer parameters from a file in the app bundle.
    
    - Parameters:
      - name: Name of the layer.
      - count: Expected number of parameters to load.
      - prefix: Added to the front of the filename, e.g. `"weights_"`
      - suffix: Added to the back of the filename, e.g. `"_W"`
      - ext: The file extension, e.g. `"bin"`
  */
  public init?(name: String, count: Int, prefix: String = "", suffix: String = "", ext: String) {
    fileSize = count * MemoryLayout<Float>.stride

    let resourceName = prefix + name + suffix
    guard let path = Bundle.main.path(forResource: resourceName, ofType: ext) else {
      print("Error: resource \"\(resourceName)\" not found")
      return nil
    }

    fd = open(path, O_RDONLY, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)
    if fd == -1 {
      print("Error: failed to open \"\(path)\", error = \(errno)")
      return nil
    }

    hdr = mmap(nil, fileSize, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0)
    if hdr == nil {
      print("Error: mmap failed, errno = \(errno)")
      return nil
    }

    pointer = hdr.bindMemory(to: Float.self, capacity: count)
    if pointer == UnsafeMutablePointer<Float>(bitPattern: -1) {
      print("Error: mmap failed, errno = \(errno)")
      return nil
    }
  }

  deinit {
    if let hdr = hdr {
      let result = munmap(hdr, Int(fileSize))
      assert(result == 0, "Error: munmap failed, errno = \(errno)")
    }
    if let fd = fd {
      close(fd)
    }
  }
}

/**
  Fills up the weights and bias arrays with random values. Useful for quickly
  trying out a model.
*/
public class ParameterLoaderRandom: ParameterData {
  private(set) public var pointer: UnsafeMutablePointer<Float>

  public init(count: Int) {
    let p = malloc(count * MemoryLayout<Float>.stride)
    pointer = p!.bindMemory(to: Float.self, capacity: count)
    Random.uniformRandom(pointer, count: count, scale: 0.1)
  }

  deinit {
    free(pointer)
  }
}

public class ParameterLoaderFromMemory: ParameterData {
    private(set) public var pointer: UnsafeMutablePointer<Float>
    
    public init?(name: String, count: Int, weights: [String : [Float]], suffix: String = "") {
        print("ParameterLoaderFromMemory: Loading params for \(name+suffix)")
        let data = weights[name+suffix]
        
        if let data : [Float] = data {
            if data.count != count {
                print("#ERROR: array size mismatch for array named:"+name)
                return nil
            }
            self.pointer = UnsafeMutablePointer(mutating: data)
        } else {
            print("#ERROR: no array named:"+name)
            return nil
        }
    }
    
    deinit {
        //free(pointer)
    }
}

// A loader that allows to load data either either from the bundle or for selective layers data from memory that
// has been modified for debug purposes
public class ParameterLoaderBundleOrMemory: ParameterData {
    private(set) public var pointer: UnsafeMutablePointer<Float>
    
    private var loader: ParameterData?
    
    public init?(name: String, count: Int, prefix: String = "", suffix: String = "", ext: String, weights: [String : [Float]]) {
        let data = weights[name+suffix]
        if data != nil {
            self.loader = ParameterLoaderFromMemory(name: name, count: count, weights: weights, suffix: suffix)
            self.pointer = loader!.pointer
        } else {
            self.loader = ParameterLoaderBundle(name: name, count: count, prefix: prefix, suffix: suffix, ext: ext)
            self.pointer = loader!.pointer
        }
    }
}

// FUTURE: add ParameterLoaderAssetCatalog to load using NSDataAsset
