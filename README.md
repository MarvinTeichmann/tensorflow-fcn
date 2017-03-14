### Update

An example on how to integrate this code into your own semantic segmentation pipeline can be found in my [KittiSeg](https://github.com/MarvinTeichmann/KittiSeg) project repository.

# tensorflow-fcn
This is a one file Tensorflow implementation of [Fully Convolutional Networks](http://arxiv.org/abs/1411.4038) in Tensorflow. The code can easily be integrated in your semantic segmentation pipeline. The network can be applied directly or finetuned to perform semantic segmentation using tensorflow training code.

Deconvolution Layers are initialized as bilinear upsampling. Conv and FCN layer weights using VGG weights. Numpy load is used to read VGG weights. No Caffe or Caffe-Tensorflow is required to run this. **The .npy file for [VGG16] to be downloaded before using this needwork**. You can find the file here: ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy

No Pascal VOC finetuning was applied to the weights. The model is meant to be finetuned on your own data. The model can be applied to an image directly (see `test_fcn32_vgg.py`) but the result will be rather coarse.

## Requirements

In addition to tensorflow the following packages are required:

numpy
scipy
pillow
matplotlib

Those packages can be installed by running `pip install -r requirements.txt` or `pip install numpy scipy pillow matplotlib`.

### Tensorflow 1.0rc

This code requires `Tensorflow Version >= 1.0rc` to run. If you want to use older Version you can try using commit `bf9400c6303826e1c25bf09a3b032e51cef57e3b`. This Commit has been tested using the pip version of `0.12`, `0.11` and `0.10`.

Tensorflow 1.0 comes with a large number of breaking api changes. If you are currently running an older tensorflow version, I would suggest creating a new `virtualenv` and install 1.0rc using:

```bash
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0rc0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
```

Above commands will install the linux version with gpu support. For other versions follow the instructions [here](https://www.tensorflow.org/versions/r1.0/get_started/os_setup).

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

### Training

Example code for training can be found in the [KittiSeg](https://github.com/MarvinTeichmann/KittiSeg) project repository.

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
