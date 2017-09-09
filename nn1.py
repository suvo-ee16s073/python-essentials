#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:53:11 2017

@author: ee16s073
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

import tensorflow as tf
sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
y_ = tf.placeholder(tf.float32)

W = tf.Variable(tf.zeros(shape = [784, 10]))
b = tf.Variable(tf.zeros(shape = [10]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(X, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.get_default_graph().get_operations()
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  #train_step.run(feed_dict={X: batch[0], y_: batch[1]})
  sess.run(train_step, {X: batch[0], y_: batch[1]})


print(sess.run(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits = y)),{y: batch[1] , y_:batch[1]}))


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={X: mnist.test.images, y_: mnist.test.labels}))
