# tensorflow-fcn
This is a Tensorflow implementation of [Fully Convolutional Networks](http://arxiv.org/abs/1411.4038) in Tensorflow. The network can be applied directly or finetuned using tensorflow training code.

Deconvolution Layers are initialized as bilinear upsampling. Conv and FCN layer weights using VGG weights. Numpy load is used to read VGG weights. No Caffe or Kaffe-Tensorflow is required to run this. The .npy file for <a href="https://dl.dropboxusercontent.com/u/50333326/vgg16.npy">VGG16</a> however need to be downloaded before using this needwork.

No Pascal VOC finetuning was applied to the weights. The model is meant to be finetuned on your own data. The model can be applied to an image directly (see `test_fcn32_vgg.py`) but the result will be rather coarse.

## Usage

`python test_fcn32_vgg.py` to test the implementation.

Use this to build the VGG object for finetuning:

```
vgg = vgg16.Vgg16()
vgg.build(images, train=True, num_classes=num_classes)
```
The `images` is a tensor with shape `[None, h, w, 3]`. Where `h` and `w` can have arbitrary size.
>Trick: the tensor can be a placeholder, a variable or even a constant.

Be aware, that influences the way `score_fr` (the original `fc8` layer) is initialized. 

### Finetuning and training

For training use `vgg.build(images, train=True, num_classes=num_classes)` were images is q queue yielding image batches.

One can use arbitrary tensorflow training code. For example the one provided by [TensorVision](https://github.com/TensorVision/TensorVision/blob/9db59e2f23755a17ddbae558f21ae371a07f1a83/tensorvision/train.py)

## Content

Currently the following Models are provided:

- FCN32

## Predecessors

Weights were generated using [Caffe to Tensorflow](https://github.com/ethereon/caffe-tensorflow). The VGG implementation is based on [tensorflow-vgg16](https://github.com/ry/tensorflow-vgg16) and numpy loading is based on [tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg). You do not need any of the above cited code to run the model, not do you need caffe.

## TODO

Provide finetuned FCN weights.