#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/10 10:45
# @Author  : Shanshan Fu
# @File    : basic.py  :
# @Contact : 33sharewithu@gmail.com



from __future__ import print_function

import tensorflow as tf


##create a session
hello = tf.constant('Hello, TensorFlow!')
# Start tf session
sess = tf.Session()
# Run the op
print(sess.run(hello))

