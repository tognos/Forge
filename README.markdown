# Preface to this (Tognos) fork of Forge

This is an extended fork that works together with [this fork of YADK](https://github.com/tognos/YAD2K) to run all Keras standard networks and the full Yolo2 model with the following performance on an iPhone 7:

| Net                      | fps  | top1  | top5  | layers | tensors | parameters  |
|--------------------------|-----:|------:|------:|-------:|--------:|------------:|
| TinyYolo  (VOC, 416x416) |   15 |     - |     - |     16 |      17 |  15.858.717 |
| Yolo 2 (COCO, 608x608)   |  2.2 |     - |     - |     30 |      32 |  50.952.553 |
| VGG16                    |  7.2 | 0.715 | 0.901 |     26 |      27 | 138.357.544 |
| Resnet-50                | 21.3 | 0.759 | 0.929 |     93 |     110 |  25.530.472 |
| Inception-V3             | 17.4 | 0.788 | 0.944 |    112 |     124 |  23.817.352 |
| InceptionResnet-V2       |  5.8 | 0.804 | 0.953 |    373 |     457 |  55.813.192 |
| Mobilenet                | 30.5 | 0.665 | 0.871 |     59 |      60 |   4.221.032 |
| Xception                 |  5.1 | 0.790 | 0.945 |    106 |     119 |  22.828.688 |

*Note: On the iPhone7, some of the networks slow down to about 30-70% of the frame rate after a minute or so which is very probably due to thermal management; pausing the app and letting the phone cool down will make it run at full frame rate again.*

This fork also features enhanced debugging capabilities:

* writes out all features maps of intermediate layers to compare them to the the original Keras feature maps

* a special mode that shows the execution, memory management and reference counts for each inference step, making it easier to pinpoint problems with complex network topologies and major changes to the toolkit

* generates a .dot file to visualize the model as graph

* More tests and a number of numpy-like functions to reshape, transpose and slice 2 to 4-dimensional swift arrays; these are mainly intended for testing and debugging and ease of use, not performance. The nicest feature is that you can write down tensor literals for test cases in swift syntax and preserve and see all the dimensions.

* some extensions to read MPSImage data in a more comprehensive way, eg. channels first oder channels last

See LayerTests and ModelTests how to use them. Setting debugTrace true in the Model-object will generate tons of debug output, showing for each step the layers being processed in each steps, along with reference counts and addresses of MPSImages annd other metadata.

In addition, this fork it has these additional layers/Tensors:

* A Collect() function that collects all argument tensors as multiple images in one MPSImage using image offsets, and a merge layer that allows calculate the sum, product, maximum and average over these collected multiple inputs, modeling the Keras Add, Multiply, Maximum and Average layers (Add is required for residual connections)

* A Space2Depth2X layer required for Yolo2

* A ZeroPadding2D layer that was required for Keras Resnet50 before the July 26th 2017 simplification but is currently no longer needed for this purpose


## Bugfixes:

* With some more complex networks the MPSTemporaryImage readcount management needed to be extended because with Concat and Collect Tensors together with branches you get not immediately obvious reads on these tensors from other branches, and together with MPSTemporaryImages shared by multiple tensors, the original readcount management just didn't work properly

* There were bugs calculating the MPSOffset for conv layers, e.g. convolutions with stride 2 and kernel size 1 did not work

* Extension for creating MPSImages from float arrays did not properly work with less than 3 channels 

## Getting started on this fork:

*  Get [this fork of YADK](https://github.com/tognos/YAD2K)
*  Follow the installation instructions for this fork
*  You can check out the YADK and Forge whereever yout want, but the commands in YADK assume they are next to each other in a common directory
*  Run the converter (see [Quickstart](https://github.com/tognos/YAD2K#quick-start))
*  Open the Forge workspace
*  Connect a suitable device running iOS 11
*  Select `ForgeTests` and build and run the tests
*  Select `Inception` and run it
*  In Inception, you can edit the init() function of class InceptionV3Network to load run another network in this app
*  In Yolo, you can edit the call `self.setNetwork(tiny: false)` in the function `createNeuralNetwork()` in CameraViewController.swift to switch between tiny yolo full 	and yolo2.
*  To convert your own Keras networks, see [here](https://github.com/tognos/YAD2K#workflow-to-add-conversion-for-a-new-network).
* Add the generated source file, the weights and eventually a test image to the testForge project first and add a test in testModels
* extend/debug the keras2metal.py and Forge if neccessary as described in the YADK readme linked abobe 
* When all works, add the weights and the builder source to your project

# Original Forge Readme, may not all apply to this fork:


# Forge: a neural network toolkit for Metal

**Forge** is a collection of helper code that makes it a little easier to construct deep neural networks using Apple's MPSCNN framework.

[Read the blog post](http://machinethink.net/blog/forge-neural-network-toolkit-for-metal/)

![Geordi likes it!](Geordi.png)

## What does this do?

Features of Forge:

**Conversion functions.** MPSCNN uses `MPSImage`s and `MTLTexture`s for everything, often using 16-bit floats. But you probably want to work with Swift `[Float]` arrays. Forge's conversion functions make it easy to work with Metal images and textures.

**Easy layer creation.** Reduce the boilerplate when building the layers for your neural network. Forge's domain-specific language makes defining a neural net as simple as:

```swift
let input = Input()

let output = input
        --> Resize(width: 28, height: 28)
        --> Convolution(kernel: (5, 5), channels: 20, activation: relu, name: "conv1")
        --> MaxPooling(kernel: (2, 2), stride: (2, 2))
        --> Convolution(kernel: (5, 5), channels: 50, activation: relu, name: "conv2")
        --> MaxPooling(kernel: (2, 2), stride: (2, 2))
        --> Dense(neurons: 320, activation: relu, name: "fc1")
        --> Dense(neurons: 10, name: "fc2")
        --> Softmax()

let model = Model(input: input, output: output)
```

**Custom layers.** MPSCNN only supports a limited number of layers, so we've added a few of our own:

- Depth-wise convolution
- Transpose channels
- Deconvolution (coming soon!)

**Preprocessing kernels.** Often you need to preprocess data before it goes into the neural network. Forge comes with a few handy kernels for this:

- SubtractMeanColor
- RGB2Gray
- RGB2BGR

**Custom compute kernels.** Many neural networks require custom compute kernels, so Forge provides helpers that make it easy to write and launch your own kernels.

**Debugging tools.** When you implement a neural network in Metal you want to make sure it actually computes the correct thing. Due to the way Metal encodes the data, inspecting the contents of the `MTLTexture` objects is not always straightforward. Forge can help with this.

**Example projects.** Forge comes with a number of pretrained neural networks, such as LeNet-5 on MNIST, Inception3 on ImageNet, and MobileNets.

> **Note:** A lot of the code in this library is still *experimental* and subject to change. Use at your own risk!

## Run the examples!

To see a demo of Forge in action, open **Forge.xcworkspace** in Xcode and run one of the example apps on your device.

You need at least Xcode 8.3 and a device with an A8 processor (iPhone 6 or better) running iOS 10 or later. You cannot build for the simulator as it does not support Metal.

The included examples are:

### MNIST

This example implements a very basic LeNet5-type neural network, trained on the MNIST dataset for handwritten digit recognition.

Run the app and point the camera at a handwritten digit (there are some images in the `Test Images` folder you can use for this) and the app will tell you what digit it is, and how sure it is of this prediction.

![MNIST example](Examples/MNIST/MNIST.jpg)

The small image in the top-left corner shows what the network sees (this is the output of the preprocessing shader that attempts to increase the contrast between black and white).

There are two targets in this project: 

- MNIST
- MNIST-DSL

They do the exact same thing, except the first one is written with straight MPSCNN code and the second one uses the Forge DSL and is therefore much easier to read.

### Inception-v3

Google's famous [Inception network](https://arxiv.org/pdf/1512.00567v3.pdf) for image classification. Point your phone at some object and the app will give you its top-5 predictions:

![Inception example](Examples/Inception/Inception.jpg)

The Inception example app is based on [Apple's sample code](https://developer.apple.com/library/content/samplecode/MetalImageRecognition/Introduction/Intro.html) but completely rewritten using the DSL. We use their learned parameters. Thanks, Apple!

### YOLO

YOLO is an object detection network. It can detect multiple objects in an image and will even tell you where they are!

![YOLO example](Examples/YOLO/YOLO.jpg)

The example app implements the Tiny YOLO network, which is not as accurate as the full version of [YOLO9000](https://pjreddie.com/darknet/yolo/) and can detect only 20 different kinds of objects.

[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by Joseph Redmon and Ali Farhadi (2016).

### MobileNets

The **MobileNets** example app is an implementation of the network architecture from the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861v1).

It works like Inception-v3 but is much faster. On the iPhone 6s it runs at 20 FPS with only moderate-to-high energy usage.

Forge uses the pretrained weights from [shicai/MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe).

## How to add Forge to your own project

Use Xcode 8.3 or better.

1. Copy the **Forge** folder into your project.
2. Use **File > Add Files to "YourProject" > Forge.xcodeproj** to add the Forge project inside your own project.
3. Drag **Products/Forge.framework** into the **Embedded Binaries** section of your project settings.
4. `import Forge` in your code.

NOTE: You cannot build for the simulator, only for "Generic iOS Device" or an actual device with arm64 architecture.

## How to use Forge

- [Creating a model with Forge](Docs/DSL.markdown)
- [Importing a model from Keras, TensorFlow, Caffe, etc](Docs/Importing.markdown)

## Where are the unit tests?

Run the **ForgeTests** app on a device.

The reason the tests are in a separate app is that Metal does not work on the simulator and Xcode can't run logic tests on the device. Catch-22.

## TODO

Forge is under active development. Here is the [list of bugs and upcoming features](Docs/TODO.markdown).

## License and credits

Forge is copyright 2016-2017 Matthijs Hollemans and is licensed under the terms of the [MIT license](LICENSE.txt).
