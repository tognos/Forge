//
//  Net.swift
//  Forge
//
//  Created by Pavel Mayer on 05.10.17.
//  Copyright © 2017 MachineThink. All rights reserved.
//

import Foundation

public protocol NetworkBuilder {
    var model  : Model { get }
    var device: MTLDevice {get}
    var name: String {get}
    init(device: MTLDevice)
    func compile(inflightBuffers: Int) -> Bool
}
