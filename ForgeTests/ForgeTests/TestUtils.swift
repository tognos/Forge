//
//  TestUtils.swift
//  ForgeTests
//
//  Created by Pavel Mayer on 28.09.17.
//  Copyright © 2017 Tognos GmbH. All rights reserved.
//

import Foundation
import Forge
import MetalPerformanceShaders

func URLinDocumentDirectory(fileName: String) -> URL {
    let documentsDirectoryURL = try! FileManager().url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
    return documentsDirectoryURL.appendingPathComponent(fileName)
}

extension UIImage {
    func saveInDocumentDirectory(fileName: String) {
        print("Saving", fileName)
        do {
            try UIImageJPEGRepresentation(self,1.0)!.write(to: URLinDocumentDirectory(fileName: fileName))
            print("Image added successfully")
        } catch {
            print(error)
        }
    }
}

extension Data {
    func saveInDocumentDirectory(fileName: String) -> Bool {
        print("Saving data in document directory as '\(fileName)'")
        do {
            try self.write(to: URLinDocumentDirectory(fileName: fileName))
            print("Data saved successfully")
            return true
        } catch {
            print(error)
            return false
        }
    }
}

// Swift can really need some polishing
protocol isFloat {}
extension Float : isFloat {}
extension Array where Element : isFloat {
    func saveInDocumentDirectory(fileName: String) -> Bool {
        return self.withUnsafeBytes({ (bufferptr) in
            let data = Data(bufferptr)
            return data.saveInDocumentDirectory(fileName: fileName)
        })
    }
}

func asJsonData(_ dictonary : Dictionary<String, Any>) -> Data? {
    do {
        let jsonData = try JSONSerialization.data(withJSONObject: dictonary, options: .prettyPrinted)
        //print(NSString(data: jsonData, encoding: 1)!)
        return jsonData
    } catch let error {
        print("Error serializing:",error)
    }
    return nil
}

extension Dictionary where Key == String, Value: Any {
    func asJson() -> Data? {
        return asJsonData(self)
    }
    func saveAsJson(fileName: String) -> Bool {
        let data = asJsonData(self)
        if data != nil {
            return data!.saveInDocumentDirectory(fileName: fileName)
        }
        return false
    }
}

public class DocumentDirectoryDumper : TensorDumper {
    let filePrefix : String
    public var shapes : [String: Any] = [:]
    public init(filePrefix: String) {
        self.filePrefix = filePrefix
    }
    
    public func dumpInputsOfConcatTensor(tensor: Tensor, imageOffset: Int) -> Bool {
        // shapeChannelsInterleaved = (numberOfImages, height, width, featureChannels)
        let rawData = tensor.image!.toFloatArrayChannelsInterleaved(fromImage: imageOffset, numImages:1)
        var iShape = tensor.image!.shapeChannelsInterleaved
        iShape.0 = 1
        let reshaped = rawData.reshaped(iShape)
        print("DocumentDirectoryDumper: Tensor \(tensor.shortId) is Concat Tensor, saving partial data as output of inputs")
        var ok = true
        for input in tensor.previous {
            
            let iSlice = sliceArray(reshaped, from: (0, 0, 0, input.destinationChannelOffset),
                                    size: (iShape.0, iShape.1, iShape.2, input.shape.channels))
            let fileName = filePrefix + "-" + input.shortId + ".floats"
            let rawSlice = flattened(iSlice)
            ok = rawSlice.saveInDocumentDirectory(fileName: fileName)
            
            if ok {
                shapes[fileName] = input.shape.asArray
            }
            print("DocumentDirectoryDumper: Saving output for Tensor \(input.shortId)")
        }
        return ok
    }
    
    public func dumpInputsOfCollectTensor(tensor: Tensor) -> Bool {
        var ok = true
        for (imageNumber, input) in tensor.previous.enumerated() {
            print("DocumentDirectoryDumper: Saving output for Tensor \(input.shortId)")
            let rawData = tensor.image!.toFloatArrayChannelsInterleaved(fromImage: input.destinationImageNumber, numImages:1)
            let fileName = filePrefix + "-" + input.shortId + ".floats"
            ok = ok && rawData.saveInDocumentDirectory(fileName: fileName)
            if ok {
                shapes[fileName] = input.shape.asArray
            }
            if input.typeName == "Concat" {
                print("DocumentDirectoryDumper: Collect-Tensor has a Concat-Tensor '\(input.shortId)' as input, saving its inputs")
                // Also dump output images of input
                ok = ok && dumpInputsOfConcatTensor(tensor: tensor, imageOffset: imageNumber)
            }
        }
        return ok
    }
    public func dump(tensor: Tensor) -> Bool {
        var ok = true
        if tensor.shape.numImages == 1 {
            let fileName = filePrefix + "-" + tensor.shortId + ".floats"
            let rawData = tensor.image!.toFloatArrayChannelsInterleaved()
            ok = rawData.saveInDocumentDirectory(fileName: fileName)
            if ok {
                shapes[fileName] = tensor.shape.asArray
            }
            if tensor.typeName == "Concat" {
                // Also dump output images of input
                ok = ok && dumpInputsOfConcatTensor(tensor: tensor, imageOffset: 0)
            }
        } else {
            // dump multiple images under the name of their outputs
            print("DocumentDirectoryDumper: Tensor \(tensor.shortId) has \(tensor.shape.numImages) images, saving as outputs of inputs")
            ok = ok && dumpInputsOfCollectTensor(tensor: tensor)
        }
        return ok
    }
}

func printLocation(_ location: String) {
    print(location)
}



