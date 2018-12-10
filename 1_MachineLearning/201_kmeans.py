#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/23 14:28
# @Author  : Shanshan Fu
# @File    : 201_kmeans.py  :
# @Contact : 33sharewithu@gmail.com


import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

## 忽略GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

## 下载MINIST 数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
full_data_x = mnist.train.images

## 模型参数
num_steps = 50 # Total steps to train
batch_size = 1024 # The number of samples per batch
k = 25 # The number of clusters
num_classes = 10 # The 10 digits
num_features = 784 # Each image is 28x28 pixels

## 建立placeholder变量
X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

## 模型
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)
(all_scores, cluster_idx, scores, cluster_centers_initialized, init_op,train_op) = kmeans.training_graph()
cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)

## 初始化所有变量
init_vars = tf.global_variables_initializer()

##start a session
sess = tf.Session()
# Run the initializer
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

# Training
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))














