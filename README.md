# tensorflow-fcn
This is a Tensorflow implementation of [Fully Convolutional Networks](http://arxiv.org/abs/1411.4038) in Tensorflow. The network can be applied directly or finetuned using tensorflow training code.

Deconvolution Layers are initialized as bilinear upsampling. Conv and FCN layer weights using VGG weights. Numpy load is used to read VGG weights. No Caffe or Caffe-Tensorflow is required to run this. <b>The .npy file for <a href="https://dl.dropboxusercontent.com/u/50333326/vgg16.npy">VGG16</a> however need to be downloaded before using this needwork.</b>

No Pascal VOC finetuning was applied to the weights. The model is meant to be finetuned on your own data. The model can be applied to an image directly (see `test_fcn32_vgg.py`) but the result will be rather coarse.

## Usage

`python test_fcn32_vgg.py` to test the implementation.

Use this to build the VGG object for finetuning:

```
vgg = vgg16.Vgg16()
vgg.build(images, train=True, num_classes=num_classes, random_init_fc8=True)
```
The `images` is a tensor with shape `[None, h, w, 3]`. Where `h` and `w` can have arbitrary size.
>Trick: the tensor can be a placeholder, a variable or even a constant.

Be aware, that `num_classes` influences the way `score_fr` (the original `fc8` layer) is initialized. For finetuning I recommend using the option `random_init_fc8=True`. 

### Finetuning and training

For training build the graph using `vgg.build(images, train=True, num_classes=num_classes)` were images is q queue yielding image batches. Use a softmax_cross_entropy loss function on top of the output of vgg.up. An Implementation of the loss function can be found in `loss.py`.

To train the graph you need an input producer and a training script. Have a look at [TensorVision](https://github.com/TensorVision/TensorVision/blob/9db59e2f23755a17ddbae558f21ae371a07f1a83/tensorvision/train.py) to see how to build those.

I had success finetuning the network using Adam Optimizer with a learning rate of `1e-6`.

## Content

Currently the following Models are provided:

- FCN32
- FCN16
- FCN8

## Remark

The deconv layer of tensorflow allows to provide a shape. The crop layer of the original implementation is therefore not needed.

I have slightly altered the naming of the upscore layer.

#### Field of View

The receptive field (also known as or `field of view`) of the provided model is: 

`( ( ( ( ( 7 ) * 2 + 6 ) * 2 + 6 ) * 2 + 6 ) * 2 + 4 ) * 2 + 4 = 404`

## Predecessors

Weights were generated using [Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow). The VGG implementation is based on [tensorflow-vgg16](https://github.com/ry/tensorflow-vgg16) and numpy loading is based on [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg). You do not need any of the above cited code to run the model, not do you need caffe.

## Install

Installing matplotlib from pip requires the following packages to be installed `libpng-dev`, `libjpeg8-dev`, `libfreetype6-dev` and `pkg-config`. On Debian, Linux Mint and Ubuntu Systems type:

`sudo apt-get install libpng-dev libjpeg8-dev libfreetype6-dev pkg-config` <br>
`pip install -r requirements.txt`

## TODO

- Provide finetuned FCN weights.
- Provide general training code