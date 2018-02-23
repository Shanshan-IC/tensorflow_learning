#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/23 11:06
# @Author  : Shanshan Fu
# @File    : 103_placeholder.py  :
# @Contact : 33sharewithu@gmail.com
'''
TensorFlow 还提供了 feed 机制,
variable 是直接传入常量或者变量，placeholder相当于设定某种数据格式类型，再将其数值传入进去
'''
import tensorflow as tf
input1 = tf.placeholder(tf.types.float32)
input2 = tf.placeholder(tf.types.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
  print sess.run([output], feed_dict={input1:[7.], input2:[2.]})