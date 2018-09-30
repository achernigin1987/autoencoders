import math
from enum import IntEnum
from typing import Optional

import tensorflow as tf


class ActivationType(IntEnum):
    NONE = 0
    RELU = 1
    LEAKY_RELU = 2


def conv_2d(inputs: tf.Tensor,
            filter_count: int,
            activation_type: ActivationType,
            layer_name: Optional[str] = None):
    """
    Convolutional layer
    :param inputs: The input tensor.
    :param filter_count: The number of filters in the convolution.
    :param activation_type: Activation type.
    :param layer_name: The layer name.
    :return:
    """
    filter_size = 3
    kernel_size = [filter_size, filter_size]

    # Activation function is initialized following He et al.
    # [Delving Deep into Rectifiers:
    #     Surpassing Human-Level Performance on ImageNet Classification]
    # Reference: https://arxiv.org/pdf/1502.01852.pdf
    weights_initializer = tf.truncated_normal_initializer(
        mean=0.0,
        stddev=math.sqrt(2.0 / ((filter_size ** 2) * filter_count)))

    if activation_type == ActivationType.NONE:
        activation_fn = None
    elif activation_type == ActivationType.LEAKY_RELU:
        activation_fn = tf.nn.leaky_relu
    elif activation_type == ActivationType.RELU:
        activation_fn = tf.nn.relu
    else:
        raise Exception(f'Bad activation type: {activation_type}')

    return tf.layers.conv2d(
        inputs=inputs,
        filters=filter_count,
        activation=activation_fn,
        kernel_initializer=weights_initializer,
        kernel_size=kernel_size,
        padding='SAME',
        name=layer_name)


def max_pool_2d(inputs: tf.Tensor, layer_name: Optional[str] = None):
    """
    Max-pooling layer
    :param inputs: The tensor over which to pool. Must have rank 4.
    :param layer_name: The layer name.
    :return: Max pooling tensor.
    """
    return tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=[2, 2],
        padding='SAME',
        name=layer_name)


def resize_layer(inputs, outputs_size, layer_name=None):
    return tf.image.resize_nearest_neighbor(
        images=inputs,
        size=outputs_size,
        name=layer_name)

