#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/12/10 11:04
# @Author  : Shanshan Fu
# @File    : basic_classification.py  :
# @Contact : 33sharewithu@gmail.com

## from tensorflow example

# TensorFlow and tf.keras
import tensorflow as tf
print(tf.__version__)
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

## 导入数据,时尚服装图片

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

## 预测的分类label
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape) # 60000 * 28 * 28,each image represented as 28 x 28 pixels:
print(len(train_labels))
print(test_images.shape)# 10000 * 28 * 28
print(len(test_labels))

## 可视化第一张图片

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

## 将像素值缩放至0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# 检查前25张图片的label
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

# 创建模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), # 将图像28*28展开成一维数组
    keras.layers.Dense(128, activation=tf.nn.relu), # 神经网络第一层是128个神经元，relu 激活函数
    keras.layers.Dense(10, activation=tf.nn.softmax) # 输出层是10个神经元的softmax，预测出10个概率得分，总和为1，表示为10个分类的概率
])

## 模型参数
model.compile(optimizer=tf.train.AdamOptimizer(), # 优化器
              loss='sparse_categorical_crossentropy', #损失函数
              metrics=['accuracy']) # 指标 - 用于监控训练和测试步骤

# 训练模型
model.fit(train_images, train_labels, epochs=5)

# 评估测试集
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# predict
predictions = model.predict(test_images)
# 显示测试集第一张图
print(predictions[0])
# 显示10个概率值中最高的分类
print(np.argmax(predictions[0]))


## 将预测值可视化
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# 我们来看看第 0 张图像、预测和预测数组。
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)

# 预测单张图片
img = test_images[0]
img = (np.expand_dims(img,0))
print(img.shape)
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
print(np.argmax(predictions_single[0]))