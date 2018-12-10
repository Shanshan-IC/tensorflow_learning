#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/10 10:39
# @Author  : Shanshan Fu
# @File    : get_mnist_data.py  :
# @Contact : 33sharewithu@gmail.com

## data in tensorflow example input_data.py

# Import MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Load data
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels

# Get the next 64 images array and labels
batch_X, batch_Y = mnist.train.next_batch(64)