#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/23 15:08
# @Author  : Shanshan Fu
# @File    : 205_regression.py  :
# @Contact : 33sharewithu@gmail.com
# 参考 http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder("float", [None, 784])#每一张图展平成784维的向量
W = tf.Variable(tf.zeros([784,10])) # 10维的证据值向量，初值都为0
b = tf.Variable(tf.zeros([10]))
# 回归模型，matmul指的是WX
y = tf.nn.softmax(tf.matmul(x,W) + b) # y是预测的分布
# 定义coss function，此处使用的是交叉熵：http://colah.github.io/posts/2015-09-Visual-Information/
y_ = tf.placeholder("float", [None,10]) # y_是实际分布
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # reduce_sum 计算张量里的元素的总和
# 用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化变量
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
# 迭代1000次
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100) # 随机100个批处理数据点
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 模型训练完后评估模型
# argmax找到的最大值1就是预测的分类结果，equal 判断和实际结果是否一致
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
