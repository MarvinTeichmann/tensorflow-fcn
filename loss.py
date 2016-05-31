"""This module provides the a softmax cross entropy loss for training FCN.

In order to train VGG first build the model and then feed apply vgg_fcn.up
to the loss. The loss function can be used in combination with any optimizer
(e.g. Adam) to finetune the whole model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def loss(hypes, logits, labels, num_classes):
    """Calculate the loss from the logits and the labels.

    Args:
      hypes: dict
          hyperparameters of the model
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.up as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.

    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        shape = [logits.get_shape()[0], num_classes]
        epsilon = tf.constant(value=hypes['solver']['epsilon'], shape=shape)
        logits = logits + epsilon
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits)
        # softmax = tf.Print(softmax, [tf.reduce_min(softmax)],
        #                     message='Checking for negative: ')
        head = hypes['arch']['weight']
        cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax), head),
                                       reduction_indices=[1])

        # TODO Check!

        # cross_entropy = tf.Print(cross_entropy, [tf.reduce_max(cross_entropy)
        # ],
        #                          message='Checking max of entropy: ')

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss
