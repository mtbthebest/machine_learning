#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
import numpy as np


graph = tf.Graph()
A = tf.placeholder(dtype=tf.int32, shape = (3,3))
B= tf.placeholder(dtype=tf.int32, shape=(3, 3))
X = tf.multiply(A,2)
Y = tf.matmul(A, B)



with tf.Session() as sess:
    a = np.array([[1,2,3],[3,4,5],[6,7,8]])
    b = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
    x = sess.run([X,Y], feed_dict={A:a, B:b})
    print np.random.random()
    print x
