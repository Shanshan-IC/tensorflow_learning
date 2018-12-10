#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/22 14:03
# @Author  : Shanshan Fu
# @File    : 102_variable.py  :
# @Contact : 33sharewithu@gmail.com
## variable是可变的张量
import tensorflow as tf

var = tf.Variable(0, name='counter')    # 定义一个名字是counter的变量

add_operation = tf.add(var, 1)
update_operation = tf.assign(var, add_operation)

with tf.Session() as sess:
    # 初始化所有的变量
    sess.run(tf.global_variables_initializer())
    for _ in range(3):
        sess.run(update_operation)
        print(sess.run(var))
