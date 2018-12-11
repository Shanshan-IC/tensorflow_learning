#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18/12/11 下午4:40
# @Author  : Shanshan Fu
# @File    : 105_tensor.py
# @Contact : 33sharewithu@gmail.com

import tensorflow as tf

# 'x' is [[1., 1.] 
#         [2., 2.]] 

# tf.reduce_mean()返回的是tensor t中各个元素的平均值。
tf.reduce_mean(x)
tf.reduce_mean(x, 0)
tf.reduce_mean(x, 1)

# tf.reduce_max()
# 计算tensor中的各个元素的最大值

# tf.reduce_all()
# 计算tensor中各个元素的逻辑和（and运算）

# tf.reduce_any()
## 逻辑或