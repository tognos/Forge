import MetalPerformanceShaders
import Forge

/**
  The Inception v3 network.
*/
class Inception3: NeuralNetwork {
  typealias Prediction = (label: String, probability: Float)

  let model: Model

  public init(device: MTLDevice, inflightBuffers: Int) {

    
    // Scale pixel values to [-1,1]
    let scale = MPSCNNNeuronLinear(device: device, a: 2, b: -1)
    let relu = MPSCNNNeuronReLU(device: device, a: 0)

    let input = Input()

    let initial = input
            --> Resize(width: 299, height: 299)
            --> Activation(scale)
            --> Convolution(kernel: (3, 3), channels: 32, stride: (2, 2), padding: .valid, activation: relu, name: "conv")
            --> Convolution(kernel: (3, 3), channels: 32, padding: .valid, activation: relu, name: "conv_1")
            --> Convolution(kernel: (3, 3), channels: 64, padding: .same, activation: relu, name: "conv_2")
            --> MaxPooling(kernel: (3, 3), stride: (2, 2))
            --> Convolution(kernel: (1, 1), channels: 80, padding: .valid, activation: relu, name: "conv_3")
            --> Convolution(kernel: (3, 3), channels: 192, padding: .valid, activation: relu, name: "conv_4")
            --> MaxPooling(kernel: (3, 3), stride: (2, 2))

    let avgPool = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same)

    let mixed0 = Concatenate([
      initial --> Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "mixed_conv"),
      initial --> Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "mixed_tower_conv")
              --> Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "mixed_tower_conv_1"),
      initial --> Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "mixed_tower_1_conv")
              --> Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "mixed_tower_1_conv_1")
              --> Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "mixed_tower_1_conv_2"),
      initial --> avgPool
              --> Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "mixed_tower_2_conv")
    ])

    let mixed1 = Concatenate([
      mixed0 --> Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "mixed_1_conv"),
      mixed0 --> Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "mixed_1_tower_conv")
             --> Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "mixed_1_tower_conv_1"),
      mixed0 --> Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "mixed_1_tower_1_conv")
             --> Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "mixed_1_tower_1_conv_1")
             --> Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "mixed_1_tower_1_conv_2"),
      mixed0 --> avgPool
             --> Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "mixed_1_tower_2_conv")
    ])

    let mixed2 = Concatenate([
      mixed1 --> Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "mixed_2_conv"),
      mixed1 --> Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "mixed_2_tower_conv")
             --> Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "mixed_2_tower_conv_1"),
      mixed1 --> Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "mixed_2_tower_1_conv")
             --> Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "mixed_2_tower_1_conv_1")
             --> Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "mixed_2_tower_1_conv_2"),
      mixed1 --> avgPool
             --> Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "mixed_2_tower_2_conv")
    ])

    let mixed3 = Concatenate([
      mixed2 --> Convolution(kernel: (3, 3), channels: 384, stride: (2, 2), padding: .valid, activation: relu, name: "mixed_3_conv"),
      mixed2 --> Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "mixed_3_tower_conv")
             --> Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "mixed_3_tower_conv_1")
             --> Convolution(kernel: (3, 3), channels: 96, stride: (2, 2), padding: .valid, activation: relu, name: "mixed_3_tower_conv_2"),
      mixed2 --> MaxPooling(kernel: (3, 3), stride: (2, 2))
    ])

    let mixed4 = Concatenate([
      mixed3 --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_4_conv"),
      mixed3 --> Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "mixed_4_tower_conv")
             --> Convolution(kernel: (7, 1), channels: 128, activation: relu, name: "mixed_4_tower_conv_1")
             --> Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "mixed_4_tower_conv_2"),
      mixed3 --> Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "mixed_4_tower_1_conv")
             --> Convolution(kernel: (1, 7), channels: 128, activation: relu, name: "mixed_4_tower_1_conv_1")
             --> Convolution(kernel: (7, 1), channels: 128, activation: relu, name: "mixed_4_tower_1_conv_2")
             --> Convolution(kernel: (1, 7), channels: 128, activation: relu, name: "mixed_4_tower_1_conv_3")
             --> Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "mixed_4_tower_1_conv_4"),
      mixed3 --> avgPool
             --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_4_tower_2_conv")
    ])

    let mixed5 = Concatenate([
      mixed4 --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_5_conv"),
      mixed4 --> Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "mixed_5_tower_conv")
             --> Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "mixed_5_tower_conv_1")
             --> Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "mixed_5_tower_conv_2"),
      mixed4 --> Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "mixed_5_tower_1_conv")
             --> Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "mixed_5_tower_1_conv_1")
             --> Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "mixed_5_tower_1_conv_2")
             --> Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "mixed_5_tower_1_conv_3")
             --> Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "mixed_5_tower_1_conv_4"),
      mixed4 --> avgPool
             --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_5_tower_2_conv")
    ])

    let mixed6 = Concatenate([
      mixed5 --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_6_conv"),
      mixed5 --> Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "mixed_6_tower_conv")
             --> Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "mixed_6_tower_conv_1")
             --> Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "mixed_6_tower_conv_2"),
      mixed5 --> Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "mixed_6_tower_1_conv")
             --> Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "mixed_6_tower_1_conv_1")
             --> Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "mixed_6_tower_1_conv_2")
             --> Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "mixed_6_tower_1_conv_3")
             --> Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "mixed_6_tower_1_conv_4"),
      mixed5 --> avgPool
             --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_6_tower_2_conv")
    ])

    let mixed7 = Concatenate([
      mixed6 --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_7_conv"),
      mixed6 --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_7_tower_conv")
             --> Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "mixed_7_tower_conv_1")
             --> Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "mixed_7_tower_conv_2"),
      mixed6 --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_7_tower_1_conv")
             --> Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "mixed_7_tower_1_conv_1")
             --> Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "mixed_7_tower_1_conv_2")
             --> Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "mixed_7_tower_1_conv_3")
             --> Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "mixed_7_tower_1_conv_4"),
      mixed6 --> avgPool
             --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_7_tower_2_conv")
    ])

    let mixed8 = Concatenate([
      mixed7 --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_8_tower_conv")
             --> Convolution(kernel: (3, 3), channels: 320, stride: (2, 2), padding: .valid, activation: relu, name: "mixed_8_tower_conv_1"),
      mixed7 --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_8_tower_1_conv")
             --> Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "mixed_8_tower_1_conv_1")
             --> Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "mixed_8_tower_1_conv_2")
             --> Convolution(kernel: (3, 3), channels: 192, stride: (2, 2), padding: .valid, activation: relu, name: "mixed_8_tower_1_conv_3"),
      mixed7 --> MaxPooling(kernel: (3, 3), stride: (2, 2))
    ])

    let tempA = mixed8 --> Convolution(kernel: (1, 1), channels: 384, activation: relu, name: "mixed_9_tower_conv")
    let tempB = mixed8 --> Convolution(kernel: (1, 1), channels: 448, activation: relu, name: "mixed_9_tower_1_conv")
                       --> Convolution(kernel: (3, 3), channels: 384, activation: relu, name: "mixed_9_tower_1_conv_1")

    let mixed9 = Concatenate([
      mixed8 --> Convolution(kernel: (1, 1), channels: 320, activation: relu, name: "mixed_9_conv"),
      tempA  --> Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "mixed_9_tower_mixed_conv"),
      tempA  --> Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "mixed_9_tower_mixed_conv_1"),
      tempB  --> Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "mixed_9_tower_1_mixed_conv"),
      tempB  --> Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "mixed_9_tower_1_mixed_conv_1"),
      mixed8 --> avgPool
             --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_9_tower_2_conv")
    ])

    let tempC = mixed9 --> Convolution(kernel: (1, 1), channels: 384, activation: relu, name: "mixed_10_tower_conv")
    let tempD = mixed9 --> Convolution(kernel: (1, 1), channels: 448, activation: relu, name: "mixed_10_tower_1_conv")
                       --> Convolution(kernel: (3, 3), channels: 384, activation: relu, name: "mixed_10_tower_1_conv_1")

    let mixed10 = Concatenate([
      mixed9 --> Convolution(kernel: (1, 1), channels: 320, activation: relu, name: "mixed_10_conv"),
      tempC  --> Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "mixed_10_tower_mixed_conv"),
      tempC  --> Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "mixed_10_tower_mixed_conv_1"),
      tempD  --> Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "mixed_10_tower_1_mixed_conv"),
      tempD  --> Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "mixed_10_tower_1_mixed_conv_1"),
      mixed9 --> MaxPooling(kernel: (3, 3), stride: (1, 1), padding: .same)
             --> Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "mixed_10_tower_2_conv")
    ])

    let output = mixed10
             --> AveragePooling(kernel: (8, 8), stride:(4, 4))
             --> Dense(neurons: 1008, name: "softmax")
             --> Softmax()

    model = Model(input: input, output: output)

    let success = model.compile(device: device, inflightBuffers: inflightBuffers) {
      name, count, type in ParameterLoaderBundle(name: name,
                                                 count: count,
                                                 prefix: type == .weights ? "weights_" : "bias_",
                                                 ext: "dat")
    }

    // begin of autogenerated forge net generation code
    
    //var model:Model
    /*
     let relu = MPSCNNNeuronReLU(device: device, a: 0)
     //let input_scale = MPSCNNNeuronLinear(device: device, a: 2, b: -1)
     let input = Input()
     let input_2 = input --> Resize(width: 299, height: 299)// --> Activation(input_scale)
     let conv2d_1 = Convolution(kernel: (3, 3), channels: 32, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_1")
     let conv2d_2 = Convolution(kernel: (3, 3), channels: 32, padding: .valid, activation: relu, name: "conv2d_2")
     let conv2d_3 = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "conv2d_3")
     let max_pooling2d_1 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_1")
     let conv2d_4 = Convolution(kernel: (1, 1), channels: 80, padding: .valid, activation: relu, name: "conv2d_4")
     let conv2d_5 = Convolution(kernel: (3, 3), channels: 192, padding: .valid, activation: relu, name: "conv2d_5")
     let max_pooling2d_2 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_2")
     let conv2d_9 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_9")
     let conv2d_7 = Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "conv2d_7")
     let conv2d_10 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_10")
     let average_pooling2d_1 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_1")
     let conv2d_6 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_6")
     let conv2d_8 = Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "conv2d_8")
     let conv2d_11 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_11")
     let conv2d_12 = Convolution(kernel: (1, 1), channels: 32, activation: relu, name: "conv2d_12")
     let conv2d_16 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_16")
     let conv2d_14 = Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "conv2d_14")
     let conv2d_17 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_17")
     let average_pooling2d_2 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_2")
     let conv2d_13 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_13")
     let conv2d_15 = Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "conv2d_15")
     let conv2d_18 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_18")
     let conv2d_19 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_19")
     let conv2d_23 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_23")
     let conv2d_21 = Convolution(kernel: (1, 1), channels: 48, activation: relu, name: "conv2d_21")
     let conv2d_24 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_24")
     let average_pooling2d_3 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_3")
     let conv2d_20 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_20")
     let conv2d_22 = Convolution(kernel: (5, 5), channels: 64, activation: relu, name: "conv2d_22")
     let conv2d_25 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_25")
     let conv2d_26 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_26")
     let conv2d_28 = Convolution(kernel: (1, 1), channels: 64, activation: relu, name: "conv2d_28")
     let conv2d_29 = Convolution(kernel: (3, 3), channels: 96, activation: relu, name: "conv2d_29")
     let conv2d_27 = Convolution(kernel: (3, 3), channels: 384, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_27")
     let conv2d_30 = Convolution(kernel: (3, 3), channels: 96, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_30")
     let max_pooling2d_3 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_3")
     let conv2d_35 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_35")
     let conv2d_36 = Convolution(kernel: (7, 1), channels: 128, activation: relu, name: "conv2d_36")
     let conv2d_32 = Convolution(kernel: (1, 1), channels: 128, activation: relu, name: "conv2d_32")
     let conv2d_37 = Convolution(kernel: (1, 7), channels: 128, activation: relu, name: "conv2d_37")
     let conv2d_33 = Convolution(kernel: (1, 7), channels: 128, activation: relu, name: "conv2d_33")
     let conv2d_38 = Convolution(kernel: (7, 1), channels: 128, activation: relu, name: "conv2d_38")
     let average_pooling2d_4 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_4")
     let conv2d_31 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_31")
     let conv2d_34 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_34")
     let conv2d_39 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_39")
     let conv2d_40 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_40")
     let conv2d_45 = Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "conv2d_45")
     let conv2d_46 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_46")
     let conv2d_42 = Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "conv2d_42")
     let conv2d_47 = Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "conv2d_47")
     let conv2d_43 = Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "conv2d_43")
     let conv2d_48 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_48")
     let average_pooling2d_5 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_5")
     let conv2d_41 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_41")
     let conv2d_44 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_44")
     let conv2d_49 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_49")
     let conv2d_50 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_50")
     let conv2d_55 = Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "conv2d_55")
     let conv2d_56 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_56")
     let conv2d_52 = Convolution(kernel: (1, 1), channels: 160, activation: relu, name: "conv2d_52")
     let conv2d_57 = Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "conv2d_57")
     let conv2d_53 = Convolution(kernel: (1, 7), channels: 160, activation: relu, name: "conv2d_53")
     let conv2d_58 = Convolution(kernel: (7, 1), channels: 160, activation: relu, name: "conv2d_58")
     let average_pooling2d_6 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_6")
     let conv2d_51 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_51")
     let conv2d_54 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_54")
     let conv2d_59 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_59")
     let conv2d_60 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_60")
     let conv2d_65 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_65")
     let conv2d_66 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_66")
     let conv2d_62 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_62")
     let conv2d_67 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_67")
     let conv2d_63 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_63")
     let conv2d_68 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_68")
     let average_pooling2d_7 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_7")
     let conv2d_61 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_61")
     let conv2d_64 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_64")
     let conv2d_69 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_69")
     let conv2d_70 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_70")
     let conv2d_73 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_73")
     let conv2d_74 = Convolution(kernel: (1, 7), channels: 192, activation: relu, name: "conv2d_74")
     let conv2d_71 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_71")
     let conv2d_75 = Convolution(kernel: (7, 1), channels: 192, activation: relu, name: "conv2d_75")
     let conv2d_72 = Convolution(kernel: (3, 3), channels: 320, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_72")
     let conv2d_76 = Convolution(kernel: (3, 3), channels: 192, stride: (2, 2), padding: .valid, activation: relu, name: "conv2d_76")
     let max_pooling2d_4 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_4")
     let conv2d_81 = Convolution(kernel: (1, 1), channels: 448, activation: relu, name: "conv2d_81")
     let conv2d_78 = Convolution(kernel: (1, 1), channels: 384, activation: relu, name: "conv2d_78")
     let conv2d_82 = Convolution(kernel: (3, 3), channels: 384, activation: relu, name: "conv2d_82")
     let conv2d_79 = Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "conv2d_79")
     let conv2d_80 = Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "conv2d_80")
     let conv2d_83 = Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "conv2d_83")
     let conv2d_84 = Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "conv2d_84")
     let average_pooling2d_8 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_8")
     let conv2d_77 = Convolution(kernel: (1, 1), channels: 320, activation: relu, name: "conv2d_77")
     let conv2d_85 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_85")
     let conv2d_90 = Convolution(kernel: (1, 1), channels: 448, activation: relu, name: "conv2d_90")
     let conv2d_87 = Convolution(kernel: (1, 1), channels: 384, activation: relu, name: "conv2d_87")
     let conv2d_91 = Convolution(kernel: (3, 3), channels: 384, activation: relu, name: "conv2d_91")
     let conv2d_88 = Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "conv2d_88")
     let conv2d_89 = Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "conv2d_89")
     let conv2d_92 = Convolution(kernel: (1, 3), channels: 384, activation: relu, name: "conv2d_92")
     let conv2d_93 = Convolution(kernel: (3, 1), channels: 384, activation: relu, name: "conv2d_93")
     let average_pooling2d_9 = AveragePooling(kernel: (3, 3), stride: (1, 1), padding: .same, name: "average_pooling2d_9")
     let conv2d_86 = Convolution(kernel: (1, 1), channels: 320, activation: relu, name: "conv2d_86")
     let conv2d_94 = Convolution(kernel: (1, 1), channels: 192, activation: relu, name: "conv2d_94")
     let avg_pool = GlobalAveragePooling(name: "avg_pool", useBias: false)
     let predictions = Dense(neurons: 1000, name: "predictions")
     
     do {
     let max_pooling2d_2 = input_2 --> conv2d_1 --> conv2d_2 --> conv2d_3 --> max_pooling2d_1 --> conv2d_4
     --> conv2d_5 --> max_pooling2d_2
     let conv2d_12 = max_pooling2d_2 --> average_pooling2d_1 --> conv2d_12
     let conv2d_8 = max_pooling2d_2 --> conv2d_7 --> conv2d_8
     let conv2d_11 = max_pooling2d_2 --> conv2d_9 --> conv2d_10 --> conv2d_11
     let conv2d_6 = max_pooling2d_2 --> conv2d_6
     let mixed0 = Concatenate([conv2d_6, conv2d_8, conv2d_11, conv2d_12])
     let conv2d_13 = mixed0 --> conv2d_13
     let conv2d_15 = mixed0 --> conv2d_14 --> conv2d_15
     let conv2d_19 = mixed0 --> average_pooling2d_2 --> conv2d_19
     let conv2d_18 = mixed0 --> conv2d_16 --> conv2d_17 --> conv2d_18
     let mixed1 = Concatenate([conv2d_13, conv2d_15, conv2d_18, conv2d_19])
     let conv2d_20 = mixed1 --> conv2d_20
     let conv2d_22 = mixed1 --> conv2d_21 --> conv2d_22
     let conv2d_25 = mixed1 --> conv2d_23 --> conv2d_24 --> conv2d_25
     let conv2d_26 = mixed1 --> average_pooling2d_3 --> conv2d_26
     let mixed2 = Concatenate([conv2d_20, conv2d_22, conv2d_25, conv2d_26])
     let conv2d_30 = mixed2 --> conv2d_28 --> conv2d_29 --> conv2d_30
     let conv2d_27 = mixed2 --> conv2d_27
     let max_pooling2d_3 = mixed2 --> max_pooling2d_3
     let mixed3 = Concatenate([conv2d_27, conv2d_30, max_pooling2d_3])
     let conv2d_31 = mixed3 --> conv2d_31
     let conv2d_40 = mixed3 --> average_pooling2d_4 --> conv2d_40
     let conv2d_34 = mixed3 --> conv2d_32 --> conv2d_33 --> conv2d_34
     let conv2d_39 = mixed3 --> conv2d_35 --> conv2d_36 --> conv2d_37 --> conv2d_38 --> conv2d_39
     
     let mixed4 = Concatenate([conv2d_31, conv2d_34, conv2d_39, conv2d_40])
     let conv2d_44 = mixed4 --> conv2d_42 --> conv2d_43 --> conv2d_44
     let conv2d_49 = mixed4 --> conv2d_45 --> conv2d_46 --> conv2d_47 --> conv2d_48 --> conv2d_49
     
     let conv2d_41 = mixed4 --> conv2d_41
     let conv2d_50 = mixed4 --> average_pooling2d_5 --> conv2d_50
     let mixed5 = Concatenate([conv2d_41, conv2d_44, conv2d_49, conv2d_50])
     let conv2d_54 = mixed5 --> conv2d_52 --> conv2d_53 --> conv2d_54
     let conv2d_59 = mixed5 --> conv2d_55 --> conv2d_56 --> conv2d_57 --> conv2d_58 --> conv2d_59
     
     let conv2d_60 = mixed5 --> average_pooling2d_6 --> conv2d_60
     let conv2d_51 = mixed5 --> conv2d_51
     let mixed6 = Concatenate([conv2d_51, conv2d_54, conv2d_59, conv2d_60])
     let conv2d_64 = mixed6 --> conv2d_62 --> conv2d_63 --> conv2d_64
     let conv2d_70 = mixed6 --> average_pooling2d_7 --> conv2d_70
     let conv2d_61 = mixed6 --> conv2d_61
     let conv2d_69 = mixed6 --> conv2d_65 --> conv2d_66 --> conv2d_67 --> conv2d_68 --> conv2d_69
     
     let mixed7 = Concatenate([conv2d_61, conv2d_64, conv2d_69, conv2d_70])
     let conv2d_72 = mixed7 --> conv2d_71 --> conv2d_72
     let max_pooling2d_4 = mixed7 --> max_pooling2d_4
     let conv2d_76 = mixed7 --> conv2d_73 --> conv2d_74 --> conv2d_75 --> conv2d_76
     let mixed8 = Concatenate([conv2d_72, conv2d_76, max_pooling2d_4])
     let conv2d_85 = mixed8 --> average_pooling2d_8 --> conv2d_85
     let conv2d_82 = mixed8 --> conv2d_81 --> conv2d_82
     let conv2d_83 = conv2d_82 --> conv2d_83
     let conv2d_78 = mixed8 --> conv2d_78
     let conv2d_84 = conv2d_82 --> conv2d_84
     let conv2d_77 = mixed8 --> conv2d_77
     let concatenate_1 = Concatenate([conv2d_83, conv2d_84])
     let conv2d_80 = conv2d_78 --> conv2d_80
     let conv2d_79 = conv2d_78 --> conv2d_79
     let mixed9_0 = Concatenate([conv2d_79, conv2d_80])
     let mixed9 = Concatenate([conv2d_77, mixed9_0, concatenate_1, conv2d_85])
     let conv2d_91 = mixed9 --> conv2d_90 --> conv2d_91
     let conv2d_87 = mixed9 --> conv2d_87
     let conv2d_94 = mixed9 --> average_pooling2d_9 --> conv2d_94
     let conv2d_86 = mixed9 --> conv2d_86
     let conv2d_92 = conv2d_91 --> conv2d_92
     let conv2d_89 = conv2d_87 --> conv2d_89
     let conv2d_93 = conv2d_91 --> conv2d_93
     let conv2d_88 = conv2d_87 --> conv2d_88
     let mixed9_1 = Concatenate([conv2d_88, conv2d_89])
     let concatenate_2 = Concatenate([conv2d_92, conv2d_93])
     let mixed10 = Concatenate([conv2d_86, mixed9_1, concatenate_2, conv2d_94])
     let predictions = mixed10 --> avg_pool --> predictions
     let output = predictions --> Softmax()
     model = Model(input: input, output: output)
     }
     let success = model.compile(device: device, inflightBuffers: inflightBuffers) {
     name, count, type in ParameterLoaderBundle(name: name,
     count: count,
     prefix: "inception_v3-",
     suffix: type == .weights ? ".weights" : ".biases",
     ext: "bin")
     }
     
     // end of autogenerated forge net generation code
     */
    
    
    /*
    // begin of autogenerated forge net generation code

    let relu = MPSCNNNeuronReLU(device: device, a: 0)
    let input = Input()
    let input_2 = input --> Resize(width: 224, height: 224)
    let zero_padding2d_1 = ZeroPadding(tblr_padding: (3, 3, 3, 3), name: "zero_padding2d_1")
    let conv1 = Convolution(kernel: (7, 7), channels: 64, stride: (2, 2), padding: .valid, activation: relu, name: "conv1")
    let max_pooling2d_1 = MaxPooling(kernel: (3, 3), stride: (2, 2), name: "max_pooling2d_1")
    let res2a_branch2a = Convolution(kernel: (1, 1), channels: 64, padding: .valid, activation: relu, name: "res2a_branch2a")
    let res2a_branch2b = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "res2a_branch2b")
    let res2a_branch2c = Convolution(kernel: (1, 1), channels: 256, padding: .valid, name: "res2a_branch2c")
    let res2a_branch1 = Convolution(kernel: (1, 1), channels: 256, padding: .valid, name: "res2a_branch1")
    let activation_4 = Activation(relu, name: "activation_4")
    let res2b_branch2a = Convolution(kernel: (1, 1), channels: 64, padding: .valid, activation: relu, name: "res2b_branch2a")
    let res2b_branch2b = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "res2b_branch2b")
    let res2b_branch2c = Convolution(kernel: (1, 1), channels: 256, padding: .valid, name: "res2b_branch2c")
    let activation_7 = Activation(relu, name: "activation_7")
    let res2c_branch2a = Convolution(kernel: (1, 1), channels: 64, padding: .valid, activation: relu, name: "res2c_branch2a")
    let res2c_branch2b = Convolution(kernel: (3, 3), channels: 64, activation: relu, name: "res2c_branch2b")
    let res2c_branch2c = Convolution(kernel: (1, 1), channels: 256, padding: .valid, name: "res2c_branch2c")
    let activation_10 = Activation(relu, name: "activation_10")
    let res3a_branch2a = Convolution(kernel: (1, 1), channels: 128, stride: (2, 2), padding: .valid, activation: relu, name: "res3a_branch2a")
    let res3a_branch2b = Convolution(kernel: (3, 3), channels: 128, activation: relu, name: "res3a_branch2b")
    let res3a_branch2c = Convolution(kernel: (1, 1), channels: 512, padding: .valid, name: "res3a_branch2c")
    let res3a_branch1 = Convolution(kernel: (1, 1), channels: 512, stride: (2, 2), padding: .valid, name: "res3a_branch1")
    let activation_13 = Activation(relu, name: "activation_13")
    let res3b_branch2a = Convolution(kernel: (1, 1), channels: 128, padding: .valid, activation: relu, name: "res3b_branch2a")
    let res3b_branch2b = Convolution(kernel: (3, 3), channels: 128, activation: relu, name: "res3b_branch2b")
    let res3b_branch2c = Convolution(kernel: (1, 1), channels: 512, padding: .valid, name: "res3b_branch2c")
    let activation_16 = Activation(relu, name: "activation_16")
    let res3c_branch2a = Convolution(kernel: (1, 1), channels: 128, padding: .valid, activation: relu, name: "res3c_branch2a")
    let res3c_branch2b = Convolution(kernel: (3, 3), channels: 128, activation: relu, name: "res3c_branch2b")
    let res3c_branch2c = Convolution(kernel: (1, 1), channels: 512, padding: .valid, name: "res3c_branch2c")
    let activation_19 = Activation(relu, name: "activation_19")
    let res3d_branch2a = Convolution(kernel: (1, 1), channels: 128, padding: .valid, activation: relu, name: "res3d_branch2a")
    let res3d_branch2b = Convolution(kernel: (3, 3), channels: 128, activation: relu, name: "res3d_branch2b")
    let res3d_branch2c = Convolution(kernel: (1, 1), channels: 512, padding: .valid, name: "res3d_branch2c")
    let activation_22 = Activation(relu, name: "activation_22")
    let res4a_branch2a = Convolution(kernel: (1, 1), channels: 256, stride: (2, 2), padding: .valid, activation: relu, name: "res4a_branch2a")
    let res4a_branch2b = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "res4a_branch2b")
    let res4a_branch2c = Convolution(kernel: (1, 1), channels: 1024, padding: .valid, name: "res4a_branch2c")
    let res4a_branch1 = Convolution(kernel: (1, 1), channels: 1024, stride: (2, 2), padding: .valid, name: "res4a_branch1")
    let activation_25 = Activation(relu, name: "activation_25")
    let res4b_branch2a = Convolution(kernel: (1, 1), channels: 256, padding: .valid, activation: relu, name: "res4b_branch2a")
    let res4b_branch2b = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "res4b_branch2b")
    let res4b_branch2c = Convolution(kernel: (1, 1), channels: 1024, padding: .valid, name: "res4b_branch2c")
    let activation_28 = Activation(relu, name: "activation_28")
    let res4c_branch2a = Convolution(kernel: (1, 1), channels: 256, padding: .valid, activation: relu, name: "res4c_branch2a")
    let res4c_branch2b = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "res4c_branch2b")
    let res4c_branch2c = Convolution(kernel: (1, 1), channels: 1024, padding: .valid, name: "res4c_branch2c")
    let activation_31 = Activation(relu, name: "activation_31")
    let res4d_branch2a = Convolution(kernel: (1, 1), channels: 256, padding: .valid, activation: relu, name: "res4d_branch2a")
    let res4d_branch2b = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "res4d_branch2b")
    let res4d_branch2c = Convolution(kernel: (1, 1), channels: 1024, padding: .valid, name: "res4d_branch2c")
    let activation_34 = Activation(relu, name: "activation_34")
    let res4e_branch2a = Convolution(kernel: (1, 1), channels: 256, padding: .valid, activation: relu, name: "res4e_branch2a")
    let res4e_branch2b = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "res4e_branch2b")
    let res4e_branch2c = Convolution(kernel: (1, 1), channels: 1024, padding: .valid, name: "res4e_branch2c")
    let activation_37 = Activation(relu, name: "activation_37")
    let res4f_branch2a = Convolution(kernel: (1, 1), channels: 256, padding: .valid, activation: relu, name: "res4f_branch2a")
    let res4f_branch2b = Convolution(kernel: (3, 3), channels: 256, activation: relu, name: "res4f_branch2b")
    let res4f_branch2c = Convolution(kernel: (1, 1), channels: 1024, padding: .valid, name: "res4f_branch2c")
    let activation_40 = Activation(relu, name: "activation_40")
    let res5a_branch2a = Convolution(kernel: (1, 1), channels: 512, stride: (2, 2), padding: .valid, activation: relu, name: "res5a_branch2a")
    let res5a_branch2b = Convolution(kernel: (3, 3), channels: 512, activation: relu, name: "res5a_branch2b")
    let res5a_branch2c = Convolution(kernel: (1, 1), channels: 2048, padding: .valid, name: "res5a_branch2c")
    let res5a_branch1 = Convolution(kernel: (1, 1), channels: 2048, stride: (2, 2), padding: .valid, name: "res5a_branch1")
    let activation_43 = Activation(relu, name: "activation_43")
    let res5b_branch2a = Convolution(kernel: (1, 1), channels: 512, padding: .valid, activation: relu, name: "res5b_branch2a")
    let res5b_branch2b = Convolution(kernel: (3, 3), channels: 512, activation: relu, name: "res5b_branch2b")
    let res5b_branch2c = Convolution(kernel: (1, 1), channels: 2048, padding: .valid, name: "res5b_branch2c")
    let activation_46 = Activation(relu, name: "activation_46")
    let res5c_branch2a = Convolution(kernel: (1, 1), channels: 512, padding: .valid, activation: relu, name: "res5c_branch2a")
    let res5c_branch2b = Convolution(kernel: (3, 3), channels: 512, activation: relu, name: "res5c_branch2b")
    let res5c_branch2c = Convolution(kernel: (1, 1), channels: 2048, padding: .valid, name: "res5c_branch2c")
    let activation_49 = Activation(relu, name: "activation_49")
    let avg_pool = AveragePooling(kernel: (7, 7), stride: (7, 7), name: "avg_pool")
    let fc1000 = Dense(neurons: 1000, name: "fc1000")
    
    do {
      let max_pooling2d_1 = input_2 --> zero_padding2d_1 --> conv1 --> max_pooling2d_1
      let res2a_branch2c = max_pooling2d_1 --> res2a_branch2a --> res2a_branch2b --> res2a_branch2c
      
      let res2a_branch1 = max_pooling2d_1 --> res2a_branch1
      let add_1 = Collect([res2a_branch2c, res2a_branch1], name: "for_add_1") --> Add(name: "add_1")
      let activation_4 = add_1 --> activation_4
      let res2b_branch2c = activation_4 --> res2b_branch2a --> res2b_branch2b --> res2b_branch2c
      let add_2 = Collect([res2b_branch2c, activation_4], name: "for_add_2") --> Add(name: "add_2")
      let activation_7 = add_2 --> activation_7
      let res2c_branch2c = activation_7 --> res2c_branch2a --> res2c_branch2b --> res2c_branch2c
      let add_3 = Collect([res2c_branch2c, activation_7], name: "for_add_3") --> Add(name: "add_3")
      let activation_10 = add_3 --> activation_10
      let res3a_branch1 = activation_10 --> res3a_branch1
      let res3a_branch2c = activation_10 --> res3a_branch2a --> res3a_branch2b --> res3a_branch2c
      let add_4 = Collect([res3a_branch2c, res3a_branch1], name: "for_add_4") --> Add(name: "add_4")
      let activation_13 = add_4 --> activation_13
      let res3b_branch2c = activation_13 --> res3b_branch2a --> res3b_branch2b --> res3b_branch2c
      let add_5 = Collect([res3b_branch2c, activation_13], name: "for_add_5") --> Add(name: "add_5")
      let activation_16 = add_5 --> activation_16
      let res3c_branch2c = activation_16 --> res3c_branch2a --> res3c_branch2b --> res3c_branch2c
      let add_6 = Collect([res3c_branch2c, activation_16], name: "for_add_6") --> Add(name: "add_6")
      let activation_19 = add_6 --> activation_19
      let res3d_branch2c = activation_19 --> res3d_branch2a --> res3d_branch2b --> res3d_branch2c
      let add_7 = Collect([res3d_branch2c, activation_19], name: "for_add_7") --> Add(name: "add_7")
      let activation_22 = add_7 --> activation_22
      let res4a_branch2c = activation_22 --> res4a_branch2a --> res4a_branch2b --> res4a_branch2c
      let res4a_branch1 = activation_22 --> res4a_branch1
      let add_8 = Collect([res4a_branch2c, res4a_branch1], name: "for_add_8") --> Add(name: "add_8")
      let activation_25 = add_8 --> activation_25
      let res4b_branch2c = activation_25 --> res4b_branch2a --> res4b_branch2b --> res4b_branch2c
      let add_9 = Collect([res4b_branch2c, activation_25], name: "for_add_9") --> Add(name: "add_9")
      let activation_28 = add_9 --> activation_28
      let res4c_branch2c = activation_28 --> res4c_branch2a --> res4c_branch2b --> res4c_branch2c
      let add_10 = Collect([res4c_branch2c, activation_28], name: "for_add_10") --> Add(name: "add_10")
      let activation_31 = add_10 --> activation_31
      let res4d_branch2c = activation_31 --> res4d_branch2a --> res4d_branch2b --> res4d_branch2c
      let add_11 = Collect([res4d_branch2c, activation_31], name: "for_add_11") --> Add(name: "add_11")
      let activation_34 = add_11 --> activation_34
      let res4e_branch2c = activation_34 --> res4e_branch2a --> res4e_branch2b --> res4e_branch2c
      let add_12 = Collect([res4e_branch2c, activation_34], name: "for_add_12") --> Add(name: "add_12")
      let activation_37 = add_12 --> activation_37
      let res4f_branch2c = activation_37 --> res4f_branch2a --> res4f_branch2b --> res4f_branch2c
      let add_13 = Collect([res4f_branch2c, activation_37], name: "for_add_13") --> Add(name: "add_13")
      let activation_40 = add_13 --> activation_40
      let res5a_branch1 = activation_40 --> res5a_branch1
      let res5a_branch2c = activation_40 --> res5a_branch2a --> res5a_branch2b --> res5a_branch2c
      let add_14 = Collect([res5a_branch2c, res5a_branch1], name: "for_add_14") --> Add(name: "add_14")
      let activation_43 = add_14 --> activation_43
      let res5b_branch2c = activation_43 --> res5b_branch2a --> res5b_branch2b --> res5b_branch2c
      let add_15 = Collect([res5b_branch2c, activation_43], name: "for_add_15") --> Add(name: "add_15")
      let activation_46 = add_15 --> activation_46
      let res5c_branch2c = activation_46 --> res5c_branch2a --> res5c_branch2b --> res5c_branch2c
      let add_16 = Collect([res5c_branch2c, activation_46], name: "for_add_16") --> Add(name: "add_16")
      let fc1000 = add_16 --> activation_49 --> avg_pool --> fc1000
      let output = fc1000 --> Softmax()
      model = Model(input: input, output: output)
    }
    let success = model.compile(device: device, inflightBuffers: inflightBuffers) {
      name, count, type in ParameterLoaderBundle(name: name,
                                                 count: count,
                                                 prefix: "resnet_50-",
                                                 suffix: type == .weights ? ".weights" : ".biases",
                                                 ext: "bin")
    }
    
    // end of autogenerated forge net generation code
    */
    
    if success {
      print(model.summary())
    }
  }

  public func encode(commandBuffer: MTLCommandBuffer, texture sourceTexture: MTLTexture, inflightIndex: Int) {
    model.encode(commandBuffer: commandBuffer, texture: sourceTexture, inflightIndex: inflightIndex)
  }

  public func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<Prediction> {
    let probabilities = model.outputImage(inflightIndex: inflightIndex).toFloatArray()
    assert(probabilities.count == 1008)

    var result = NeuralNetworkResult<Prediction>()
    result.predictions = probabilities.top(k: 5).map { x -> Prediction in (self.labels[x.0], x.1) }
    return result
  }

  let labels = [
    "",
    "kit fox, Vulpes macrotis",
    "English setter",
    "Siberian husky",
    "Australian terrier",
    "English springer, English springer spaniel",
    "grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus",
    "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens",
    "Egyptian cat",
    "ibex, Capra ibex",
    "Persian cat",
    "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor",
    "gazelle",
    "porcupine, hedgehog",
    "sea lion",
    "malamute, malemute, Alaskan malamute",
    "badger",
    "Great Dane",
    "Walker hound, Walker foxhound",
    "Welsh springer spaniel",
    "whippet",
    "Scottish deerhound, deerhound",
    "killer whale, killer, orca, grampus, sea wolf, Orcinus orca",
    "mink",
    "African elephant, Loxodonta africana",
    "Weimaraner",
    "soft-coated wheaten terrier",
    "Dandie Dinmont, Dandie Dinmont terrier",
    "red wolf, maned wolf, Canis rufus, Canis niger",
    "Old English sheepdog, bobtail",
    "jaguar, panther, Panthera onca, Felis onca",
    "otterhound, otter hound",
    "bloodhound, sleuthhound",
    "Airedale, Airedale terrier",
    "hyena, hyaena",
    "meerkat, mierkat",
    "giant schnauzer",
    "titi, titi monkey",
    "three-toed sloth, ai, Bradypus tridactylus",
    "sorrel",
    "black-footed ferret, ferret, Mustela nigripes",
    "dalmatian, coach dog, carriage dog",
    "black-and-tan coonhound",
    "papillon",
    "skunk, polecat, wood pussy",
    "Staffordshire bullterrier, Staffordshire bull terrier",
    "Mexican hairless",
    "Bouvier des Flandres, Bouviers des Flandres",
    "weasel",
    "miniature poodle",
    "Cardigan, Cardigan Welsh corgi",
    "malinois",
    "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis",
    "fox squirrel, eastern fox squirrel, Sciurus niger",
    "colobus, colobus monkey",
    "tiger cat",
    "Lhasa, Lhasa apso",
    "impala, Aepyceros melampus",
    "coyote, prairie wolf, brush wolf, Canis latrans",
    "Yorkshire terrier",
    "Newfoundland, Newfoundland dog",
    "brown bear, bruin, Ursus arctos",
    "red fox, Vulpes vulpes",
    "Norwegian elkhound, elkhound",
    "Rottweiler",
    "hartebeest",
    "Saluki, gazelle hound",
    "grey fox, gray fox, Urocyon cinereoargenteus",
    "schipperke",
    "Pekinese, Pekingese, Peke",
    "Brabancon griffon",
    "West Highland white terrier",
    "Sealyham terrier, Sealyham",
    "guenon, guenon monkey",
    "mongoose",
    "indri, indris, Indri indri, Indri brevicaudatus",
    "tiger, Panthera tigris",
    "Irish wolfhound",
    "wild boar, boar, Sus scrofa",
    "EntleBucher",
    "zebra",
    "ram, tup",
    "French bulldog",
    "orangutan, orang, orangutang, Pongo pygmaeus",
    "basenji",
    "leopard, Panthera pardus",
    "Bernese mountain dog",
    "Maltese dog, Maltese terrier, Maltese",
    "Norfolk terrier",
    "toy terrier",
    "vizsla, Hungarian pointer",
    "cairn, cairn terrier",
    "squirrel monkey, Saimiri sciureus",
    "groenendael",
    "clumber, clumber spaniel",
    "Siamese cat, Siamese",
    "chimpanzee, chimp, Pan troglodytes",
    "komondor",
    "Afghan hound, Afghan",
    "Japanese spaniel",
    "proboscis monkey, Nasalis larvatus",
    "guinea pig, Cavia cobaya",
    "white wolf, Arctic wolf, Canis lupus tundrarum",
    "ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus",
    "gorilla, Gorilla gorilla",
    "borzoi, Russian wolfhound",
    "toy poodle",
    "Kerry blue terrier",
    "ox",
    "Scotch terrier, Scottish terrier, Scottie",
    "Tibetan mastiff",
    "spider monkey, Ateles geoffroyi",
    "Doberman, Doberman pinscher",
    "Boston bull, Boston terrier",
    "Greater Swiss Mountain dog",
    "Appenzeller",
    "Shih-Tzu",
    "Irish water spaniel",
    "Pomeranian",
    "Bedlington terrier",
    "warthog",
    "Arabian camel, dromedary, Camelus dromedarius",
    "siamang, Hylobates syndactylus, Symphalangus syndactylus",
    "miniature schnauzer",
    "collie",
    "golden retriever",
    "Irish terrier",
    "affenpinscher, monkey pinscher, monkey dog",
    "Border collie",
    "hare",
    "boxer",
    "silky terrier, Sydney silky",
    "beagle",
    "Leonberg",
    "German short-haired pointer",
    "patas, hussar monkey, Erythrocebus patas",
    "dhole, Cuon alpinus",
    "baboon",
    "macaque",
    "Chesapeake Bay retriever",
    "bull mastiff",
    "kuvasz",
    "capuchin, ringtail, Cebus capucinus",
    "pug, pug-dog",
    "curly-coated retriever",
    "Norwich terrier",
    "flat-coated retriever",
    "hog, pig, grunter, squealer, Sus scrofa",
    "keeshond",
    "Eskimo dog, husky",
    "Brittany spaniel",
    "standard poodle",
    "Lakeland terrier",
    "snow leopard, ounce, Panthera uncia",
    "Gordon setter",
    "dingo, warrigal, warragal, Canis dingo",
    "standard schnauzer",
    "hamster",
    "Tibetan terrier, chrysanthemum dog",
    "Arctic fox, white fox, Alopex lagopus",
    "wire-haired fox terrier",
    "basset, basset hound",
    "water buffalo, water ox, Asiatic buffalo, Bubalus bubalis",
    "American black bear, black bear, Ursus americanus, Euarctos americanus",
    "Angora, Angora rabbit",
    "bison",
    "howler monkey, howler",
    "hippopotamus, hippo, river horse, Hippopotamus amphibius",
    "chow, chow chow",
    "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca",
    "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier",
    "Shetland sheepdog, Shetland sheep dog, Shetland",
    "Great Pyrenees",
    "Chihuahua",
    "tabby, tabby cat",
    "marmoset",
    "Labrador retriever",
    "Saint Bernard, St Bernard",
    "armadillo",
    "Samoyed, Samoyede",
    "bluetick",
    "redbone",
    "polecat, fitch, foulmart, foumart, Mustela putorius",
    "marmot",
    "kelpie",
    "gibbon, Hylobates lar",
    "llama",
    "miniature pinscher",
    "wood rabbit, cottontail, cottontail rabbit",
    "Italian greyhound",
    "lion, king of beasts, Panthera leo",
    "cocker spaniel, English cocker spaniel, cocker",
    "Irish setter, red setter",
    "dugong, Dugong dugon",
    "Indian elephant, Elephas maximus",
    "beaver",
    "Sussex spaniel",
    "Pembroke, Pembroke Welsh corgi",
    "Blenheim spaniel",
    "Madagascar cat, ring-tailed lemur, Lemur catta",
    "Rhodesian ridgeback",
    "lynx, catamount",
    "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus",
    "langur",
    "Ibizan hound, Ibizan Podenco",
    "timber wolf, grey wolf, gray wolf, Canis lupus",
    "cheetah, chetah, Acinonyx jubatus",
    "English foxhound",
    "briard",
    "sloth bear, Melursus ursinus, Ursus ursinus",
    "Border terrier",
    "German shepherd, German shepherd dog, German police dog, alsatian",
    "otter",
    "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus",
    "tusker",
    "echidna, spiny anteater, anteater",
    "wallaby, brush kangaroo",
    "platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus",
    "wombat",
    "revolver, six-gun, six-shooter",
    "umbrella",
    "schooner",
    "soccer ball",
    "accordion, piano accordion, squeeze box",
    "ant, emmet, pismire",
    "starfish, sea star",
    "chambered nautilus, pearly nautilus, nautilus",
    "grand piano, grand",
    "laptop, laptop computer",
    "strawberry",
    "airliner",
    "warplane, military plane",
    "airship, dirigible",
    "balloon",
    "space shuttle",
    "fireboat",
    "gondola",
    "speedboat",
    "lifeboat",
    "canoe",
    "yawl",
    "catamaran",
    "trimaran",
    "container ship, containership, container vessel",
    "liner, ocean liner",
    "pirate, pirate ship",
    "aircraft carrier, carrier, flattop, attack aircraft carrier",
    "submarine, pigboat, sub, U-boat",
    "wreck",
    "half track",
    "tank, army tank, armored combat vehicle, armoured combat vehicle",
    "missile",
    "bobsled, bobsleigh, bob",
    "dogsled, dog sled, dog sleigh",
    "bicycle-built-for-two, tandem bicycle, tandem",
    "mountain bike, all-terrain bike, off-roader",
    "freight car",
    "passenger car, coach, carriage",
    "barrow, garden cart, lawn cart, wheelbarrow",
    "shopping cart",
    "motor scooter, scooter",
    "forklift",
    "electric locomotive",
    "steam locomotive",
    "amphibian, amphibious vehicle",
    "ambulance",
    "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon",
    "cab, hack, taxi, taxicab",
    "convertible",
    "jeep, landrover",
    "limousine, limo",
    "minivan",
    "Model T",
    "racer, race car, racing car",
    "sports car, sport car",
    "go-kart",
    "golfcart, golf cart",
    "moped",
    "snowplow, snowplough",
    "fire engine, fire truck",
    "garbage truck, dustcart",
    "pickup, pickup truck",
    "tow truck, tow car, wrecker",
    "trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi",
    "moving van",
    "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria",
    "recreational vehicle, RV, R.V.",
    "streetcar, tram, tramcar, trolley, trolley car",
    "snowmobile",
    "tractor",
    "mobile home, manufactured home",
    "tricycle, trike, velocipede",
    "unicycle, monocycle",
    "horse cart, horse-cart",
    "jinrikisha, ricksha, rickshaw",
    "oxcart",
    "bassinet",
    "cradle",
    "crib, cot",
    "four-poster",
    "bookcase",
    "china cabinet, china closet",
    "medicine chest, medicine cabinet",
    "chiffonier, commode",
    "table lamp",
    "file, file cabinet, filing cabinet",
    "park bench",
    "barber chair",
    "throne",
    "folding chair",
    "rocking chair, rocker",
    "studio couch, day bed",
    "toilet seat",
    "desk",
    "pool table, billiard table, snooker table",
    "dining table, board",
    "entertainment center",
    "wardrobe, closet, press",
    "Granny Smith",
    "orange",
    "lemon",
    "fig",
    "pineapple, ananas",
    "banana",
    "jackfruit, jak, jack",
    "custard apple",
    "pomegranate",
    "acorn",
    "hip, rose hip, rosehip",
    "ear, spike, capitulum",
    "rapeseed",
    "corn",
    "buckeye, horse chestnut, conker",
    "organ, pipe organ",
    "upright, upright piano",
    "chime, bell, gong",
    "drum, membranophone, tympan",
    "gong, tam-tam",
    "maraca",
    "marimba, xylophone",
    "steel drum",
    "banjo",
    "cello, violoncello",
    "violin, fiddle",
    "harp",
    "acoustic guitar",
    "electric guitar",
    "cornet, horn, trumpet, trump",
    "French horn, horn",
    "trombone",
    "harmonica, mouth organ, harp, mouth harp",
    "ocarina, sweet potato",
    "panpipe, pandean pipe, syrinx",
    "bassoon",
    "oboe, hautboy, hautbois",
    "sax, saxophone",
    "flute, transverse flute",
    "daisy",
    "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
    "cliff, drop, drop-off",
    "valley, vale",
    "alp",
    "volcano",
    "promontory, headland, head, foreland",
    "sandbar, sand bar",
    "coral reef",
    "lakeside, lakeshore",
    "seashore, coast, seacoast, sea-coast",
    "geyser",
    "hatchet",
    "cleaver, meat cleaver, chopper",
    "letter opener, paper knife, paperknife",
    "plane, carpenter's plane, woodworking plane",
    "power drill",
    "lawn mower, mower",
    "hammer",
    "corkscrew, bottle screw",
    "can opener, tin opener",
    "plunger, plumber's helper",
    "screwdriver",
    "shovel",
    "plow, plough",
    "chain saw, chainsaw",
    "cock",
    "hen",
    "ostrich, Struthio camelus",
    "brambling, Fringilla montifringilla",
    "goldfinch, Carduelis carduelis",
    "house finch, linnet, Carpodacus mexicanus",
    "junco, snowbird",
    "indigo bunting, indigo finch, indigo bird, Passerina cyanea",
    "robin, American robin, Turdus migratorius",
    "bulbul",
    "jay",
    "magpie",
    "chickadee",
    "water ouzel, dipper",
    "kite",
    "bald eagle, American eagle, Haliaeetus leucocephalus",
    "vulture",
    "great grey owl, great gray owl, Strix nebulosa",
    "black grouse",
    "ptarmigan",
    "ruffed grouse, partridge, Bonasa umbellus",
    "prairie chicken, prairie grouse, prairie fowl",
    "peacock",
    "quail",
    "partridge",
    "African grey, African gray, Psittacus erithacus",
    "macaw",
    "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita",
    "lorikeet",
    "coucal",
    "bee eater",
    "hornbill",
    "hummingbird",
    "jacamar",
    "toucan",
    "drake",
    "red-breasted merganser, Mergus serrator",
    "goose",
    "black swan, Cygnus atratus",
    "white stork, Ciconia ciconia",
    "black stork, Ciconia nigra",
    "spoonbill",
    "flamingo",
    "American egret, great white heron, Egretta albus",
    "little blue heron, Egretta caerulea",
    "bittern",
    "crane",
    "limpkin, Aramus pictus",
    "American coot, marsh hen, mud hen, water hen, Fulica americana",
    "bustard",
    "ruddy turnstone, Arenaria interpres",
    "red-backed sandpiper, dunlin, Erolia alpina",
    "redshank, Tringa totanus",
    "dowitcher",
    "oystercatcher, oyster catcher",
    "European gallinule, Porphyrio porphyrio",
    "pelican",
    "king penguin, Aptenodytes patagonica",
    "albatross, mollymawk",
    "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias",
    "tiger shark, Galeocerdo cuvieri",
    "hammerhead, hammerhead shark",
    "electric ray, crampfish, numbfish, torpedo",
    "stingray",
    "barracouta, snoek",
    "coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch",
    "tench, Tinca tinca",
    "goldfish, Carassius auratus",
    "eel",
    "rock beauty, Holocanthus tricolor",
    "anemone fish",
    "lionfish",
    "puffer, pufferfish, blowfish, globefish",
    "sturgeon",
    "gar, garfish, garpike, billfish, Lepisosteus osseus",
    "loggerhead, loggerhead turtle, Caretta caretta",
    "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea",
    "mud turtle",
    "terrapin",
    "box turtle, box tortoise",
    "banded gecko",
    "common iguana, iguana, Iguana iguana",
    "American chameleon, anole, Anolis carolinensis",
    "whiptail, whiptail lizard",
    "agama",
    "frilled lizard, Chlamydosaurus kingi",
    "alligator lizard",
    "Gila monster, Heloderma suspectum",
    "green lizard, Lacerta viridis",
    "African chameleon, Chamaeleo chamaeleon",
    "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis",
    "triceratops",
    "African crocodile, Nile crocodile, Crocodylus niloticus",
    "American alligator, Alligator mississipiensis",
    "thunder snake, worm snake, Carphophis amoenus",
    "ringneck snake, ring-necked snake, ring snake",
    "hognose snake, puff adder, sand viper",
    "green snake, grass snake",
    "king snake, kingsnake",
    "garter snake, grass snake",
    "water snake",
    "vine snake",
    "night snake, Hypsiglena torquata",
    "boa constrictor, Constrictor constrictor",
    "rock python, rock snake, Python sebae",
    "Indian cobra, Naja naja",
    "green mamba",
    "sea snake",
    "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus",
    "diamondback, diamondback rattlesnake, Crotalus adamanteus",
    "sidewinder, horned rattlesnake, Crotalus cerastes",
    "European fire salamander, Salamandra salamandra",
    "common newt, Triturus vulgaris",
    "eft",
    "spotted salamander, Ambystoma maculatum",
    "axolotl, mud puppy, Ambystoma mexicanum",
    "bullfrog, Rana catesbeiana",
    "tree frog, tree-frog",
    "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui",
    "whistle",
    "wing",
    "paintbrush",
    "hand blower, blow dryer, blow drier, hair dryer, hair drier",
    "oxygen mask",
    "snorkel",
    "loudspeaker, speaker, speaker unit, loudspeaker system, speaker system",
    "microphone, mike",
    "screen, CRT screen",
    "mouse, computer mouse",
    "electric fan, blower",
    "oil filter",
    "strainer",
    "space heater",
    "stove",
    "guillotine",
    "barometer",
    "rule, ruler",
    "odometer, hodometer, mileometer, milometer",
    "scale, weighing machine",
    "analog clock",
    "digital clock",
    "wall clock",
    "hourglass",
    "sundial",
    "parking meter",
    "stopwatch, stop watch",
    "digital watch",
    "stethoscope",
    "syringe",
    "magnetic compass",
    "binoculars, field glasses, opera glasses",
    "projector",
    "sunglasses, dark glasses, shades",
    "loupe, jeweler's loupe",
    "radio telescope, radio reflector",
    "bow",
    "cannon",
    "assault rifle, assault gun",
    "rifle",
    "projectile, missile",
    "computer keyboard, keypad",
    "typewriter keyboard",
    "crane",
    "lighter, light, igniter, ignitor",
    "abacus",
    "cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM",
    "slide rule, slipstick",
    "desktop computer",
    "hand-held computer, hand-held microcomputer",
    "notebook, notebook computer",
    "web site, website, internet site, site",
    "harvester, reaper",
    "thresher, thrasher, threshing machine",
    "printer",
    "slot, one-armed bandit",
    "vending machine",
    "sewing machine",
    "joystick",
    "switch, electric switch, electrical switch",
    "hook, claw",
    "car wheel",
    "paddlewheel, paddle wheel",
    "pinwheel",
    "potter's wheel",
    "gas pump, gasoline pump, petrol pump, island dispenser",
    "carousel, carrousel, merry-go-round, roundabout, whirligig",
    "swing",
    "reel",
    "radiator",
    "puck, hockey puck",
    "hard disc, hard disk, fixed disk",
    "sunglass",
    "pick, plectrum, plectron",
    "car mirror",
    "solar dish, solar collector, solar furnace",
    "remote control, remote",
    "disk brake, disc brake",
    "buckle",
    "hair slide",
    "knot",
    "combination lock",
    "padlock",
    "nail",
    "safety pin",
    "screw",
    "muzzle",
    "seat belt, seatbelt",
    "ski",
    "candle, taper, wax light",
    "jack-o'-lantern",
    "spotlight, spot",
    "torch",
    "neck brace",
    "pier",
    "tripod",
    "maypole",
    "mousetrap",
    "spider web, spider's web",
    "trilobite",
    "harvestman, daddy longlegs, Phalangium opilio",
    "scorpion",
    "black and gold garden spider, Argiope aurantia",
    "barn spider, Araneus cavaticus",
    "garden spider, Aranea diademata",
    "black widow, Latrodectus mactans",
    "tarantula",
    "wolf spider, hunting spider",
    "tick",
    "centipede",
    "isopod",
    "Dungeness crab, Cancer magister",
    "rock crab, Cancer irroratus",
    "fiddler crab",
    "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica",
    "American lobster, Northern lobster, Maine lobster, Homarus americanus",
    "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish",
    "crayfish, crawfish, crawdad, crawdaddy",
    "hermit crab",
    "tiger beetle",
    "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle",
    "ground beetle, carabid beetle",
    "long-horned beetle, longicorn, longicorn beetle",
    "leaf beetle, chrysomelid",
    "dung beetle",
    "rhinoceros beetle",
    "weevil",
    "fly",
    "bee",
    "grasshopper, hopper",
    "cricket",
    "walking stick, walkingstick, stick insect",
    "cockroach, roach",
    "mantis, mantid",
    "cicada, cicala",
    "leafhopper",
    "lacewing, lacewing fly",
    "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    "damselfly",
    "admiral",
    "ringlet, ringlet butterfly",
    "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus",
    "cabbage butterfly",
    "sulphur butterfly, sulfur butterfly",
    "lycaenid, lycaenid butterfly",
    "jellyfish",
    "sea anemone, anemone",
    "brain coral",
    "flatworm, platyhelminth",
    "nematode, nematode worm, roundworm",
    "conch",
    "snail",
    "slug",
    "sea slug, nudibranch",
    "chiton, coat-of-mail shell, sea cradle, polyplacophore",
    "sea urchin",
    "sea cucumber, holothurian",
    "iron, smoothing iron",
    "espresso maker",
    "microwave, microwave oven",
    "Dutch oven",
    "rotisserie",
    "toaster",
    "waffle iron",
    "vacuum, vacuum cleaner",
    "dishwasher, dish washer, dishwashing machine",
    "refrigerator, icebox",
    "washer, automatic washer, washing machine",
    "Crock Pot",
    "frying pan, frypan, skillet",
    "wok",
    "caldron, cauldron",
    "coffeepot",
    "teapot",
    "spatula",
    "altar",
    "triumphal arch",
    "patio, terrace",
    "steel arch bridge",
    "suspension bridge",
    "viaduct",
    "barn",
    "greenhouse, nursery, glasshouse",
    "palace",
    "monastery",
    "library",
    "apiary, bee house",
    "boathouse",
    "church, church building",
    "mosque",
    "stupa, tope",
    "planetarium",
    "restaurant, eating house, eating place, eatery",
    "cinema, movie theater, movie theatre, movie house, picture palace",
    "home theater, home theatre",
    "lumbermill, sawmill",
    "coil, spiral, volute, whorl, helix",
    "obelisk",
    "totem pole",
    "castle",
    "prison, prison house",
    "grocery store, grocery, food market, market",
    "bakery, bakeshop, bakehouse",
    "barbershop",
    "bookshop, bookstore, bookstall",
    "butcher shop, meat market",
    "confectionery, confectionary, candy store",
    "shoe shop, shoe-shop, shoe store",
    "tobacco shop, tobacconist shop, tobacconist",
    "toyshop",
    "fountain",
    "cliff dwelling",
    "yurt",
    "dock, dockage, docking facility",
    "brass, memorial tablet, plaque",
    "megalith, megalithic structure",
    "bannister, banister, balustrade, balusters, handrail",
    "breakwater, groin, groyne, mole, bulwark, seawall, jetty",
    "dam, dike, dyke",
    "chainlink fence",
    "picket fence, paling",
    "worm fence, snake fence, snake-rail fence, Virginia fence",
    "stone wall",
    "grille, radiator grille",
    "sliding door",
    "turnstile",
    "mountain tent",
    "scoreboard",
    "honeycomb",
    "plate rack",
    "pedestal, plinth, footstall",
    "beacon, lighthouse, beacon light, pharos",
    "mashed potato",
    "bell pepper",
    "head cabbage",
    "broccoli",
    "cauliflower",
    "zucchini, courgette",
    "spaghetti squash",
    "acorn squash",
    "butternut squash",
    "cucumber, cuke",
    "artichoke, globe artichoke",
    "cardoon",
    "mushroom",
    "shower curtain",
    "jean, blue jean, denim",
    "carton",
    "handkerchief, hankie, hanky, hankey",
    "sandal",
    "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
    "safe",
    "plate",
    "necklace",
    "croquet ball",
    "fur coat",
    "thimble",
    "pajama, pyjama, pj's, jammies",
    "running shoe",
    "cocktail shaker",
    "chest",
    "manhole cover",
    "modem",
    "tub, vat",
    "tray",
    "balance beam, beam",
    "bagel, beigel",
    "prayer rug, prayer mat",
    "kimono",
    "hot pot, hotpot",
    "whiskey jug",
    "knee pad",
    "book jacket, dust cover, dust jacket, dust wrapper",
    "spindle",
    "ski mask",
    "beer bottle",
    "crash helmet",
    "bottlecap",
    "tile roof",
    "mask",
    "maillot",
    "Petri dish",
    "football helmet",
    "bathing cap, swimming cap",
    "teddy, teddy bear",
    "holster",
    "pop bottle, soda bottle",
    "photocopier",
    "vestment",
    "crossword puzzle, crossword",
    "golf ball",
    "trifle",
    "suit, suit of clothes",
    "water tower",
    "feather boa, boa",
    "cloak",
    "red wine",
    "drumstick",
    "shield, buckler",
    "Christmas stocking",
    "hoopskirt, crinoline",
    "menu",
    "stage",
    "bonnet, poke bonnet",
    "meat loaf, meatloaf",
    "baseball",
    "face powder",
    "scabbard",
    "sunscreen, sunblock, sun blocker",
    "beer glass",
    "hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa",
    "guacamole",
    "lampshade, lamp shade",
    "wool, woolen, woollen",
    "hay",
    "bow tie, bow-tie, bowtie",
    "mailbag, postbag",
    "water jug",
    "bucket, pail",
    "dishrag, dishcloth",
    "soup bowl",
    "eggnog",
    "mortar",
    "trench coat",
    "paddle, boat paddle",
    "chain",
    "swab, swob, mop",
    "mixing bowl",
    "potpie",
    "wine bottle",
    "shoji",
    "bulletproof vest",
    "drilling platform, offshore rig",
    "binder, ring-binder",
    "cardigan",
    "sweatshirt",
    "pot, flowerpot",
    "birdhouse",
    "hamper",
    "ping-pong ball",
    "pencil box, pencil case",
    "pay-phone, pay-station",
    "consomme",
    "apron",
    "punching bag, punch bag, punching ball, punchball",
    "backpack, back pack, knapsack, packsack, rucksack, haversack",
    "groom, bridegroom",
    "bearskin, busby, shako",
    "pencil sharpener",
    "broom",
    "mosquito net",
    "abaya",
    "mortarboard",
    "poncho",
    "crutch",
    "Polaroid camera, Polaroid Land camera",
    "space bar",
    "cup",
    "racket, racquet",
    "traffic light, traffic signal, stoplight",
    "quill, quill pen",
    "radio, wireless",
    "dough",
    "cuirass",
    "military uniform",
    "lipstick, lip rouge",
    "shower cap",
    "monitor",
    "oscilloscope, scope, cathode-ray oscilloscope, CRO",
    "mitten",
    "brassiere, bra, bandeau",
    "French loaf",
    "vase",
    "milk can",
    "rugby ball",
    "paper towel",
    "earthstar",
    "envelope",
    "miniskirt, mini",
    "cowboy hat, ten-gallon hat",
    "trolleybus, trolley coach, trackless trolley",
    "perfume, essence",
    "bathtub, bathing tub, bath, tub",
    "hotdog, hot dog, red hot",
    "coral fungus",
    "bullet train, bullet",
    "pillow",
    "toilet tissue, toilet paper, bathroom tissue",
    "cassette",
    "carpenter's kit, tool kit",
    "ladle",
    "stinkhorn, carrion fungus",
    "lotion",
    "hair spray",
    "academic gown, academic robe, judge's robe",
    "dome",
    "crate",
    "wig",
    "burrito",
    "pill bottle",
    "chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour",
    "theater curtain, theatre curtain",
    "window shade",
    "barrel, cask",
    "washbasin, handbasin, washbowl, lavabo, wash-hand basin",
    "ballpoint, ballpoint pen, ballpen, Biro",
    "basketball",
    "bath towel",
    "cowboy boot",
    "gown",
    "window screen",
    "agaric",
    "cellular telephone, cellular phone, cellphone, cell, mobile phone",
    "nipple",
    "barbell",
    "mailbox, letter box",
    "lab coat, laboratory coat",
    "fire screen, fireguard",
    "minibus",
    "packet",
    "maze, labyrinth",
    "pole",
    "horizontal bar, high bar",
    "sombrero",
    "pickelhaube",
    "rain barrel",
    "wallet, billfold, notecase, pocketbook",
    "cassette player",
    "comic book",
    "piggy bank, penny bank",
    "street sign",
    "bell cote, bell cot",
    "fountain pen",
    "Windsor tie",
    "volleyball",
    "overskirt",
    "sarong",
    "purse",
    "bolo tie, bolo, bola tie, bola",
    "bib",
    "parachute, chute",
    "sleeping bag",
    "television, television system",
    "swimming trunks, bathing trunks",
    "measuring cup",
    "espresso",
    "pizza, pizza pie",
    "breastplate, aegis, egis",
    "shopping basket",
    "wooden spoon",
    "saltshaker, salt shaker",
    "chocolate sauce, chocolate syrup",
    "ballplayer, baseball player",
    "goblet",
    "gyromitra",
    "stretcher",
    "water bottle",
    "dial telephone, dial phone",
    "soap dispenser",
    "jersey, T-shirt, tee shirt",
    "school bus",
    "jigsaw puzzle",
    "plastic bag",
    "reflex camera",
    "diaper, nappy, napkin",
    "Band Aid",
    "ice lolly, lolly, lollipop, popsicle",
    "velvet",
    "tennis ball",
    "gasmask, respirator, gas helmet",
    "doormat, welcome mat",
    "Loafer",
    "ice cream, icecream",
    "pretzel",
    "quilt, comforter, comfort, puff",
    "maillot, tank suit",
    "tape player",
    "clog, geta, patten, sabot",
    "iPod",
    "bolete",
    "scuba diver",
    "pitcher, ewer",
    "matchstick",
    "bikini, two-piece",
    "sock",
    "CD player",
    "lens cap, lens cover",
    "thatch, thatched roof",
    "vault",
    "beaker",
    "bubble",
    "cheeseburger",
    "parallel bars, bars",
    "flagpole, flagstaff",
    "coffee mug",
    "rubber eraser, rubber, pencil eraser",
    "stole",
    "carbonara",
    "dumbbell"]
}
