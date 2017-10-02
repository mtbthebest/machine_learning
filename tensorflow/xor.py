#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
import numpy as np
from random import choice
from collections import OrderedDict

x_input = np.array([[0,0], [0,1], [1,0], [1,1]])
y_input = np.array([[0], [1], [1], [0]])

param = zip(x_input, y_input)
print param[3][0]
x = tf.placeholder(dtype= tf.int32, shape=x_input.shape)
W = tf.Variable(tf.zeros([2,1]),trainable=True)
b = tf.Variable(0.0, trainable=True)


init = tf.global_variables_initializer()
mul = tf.matmul(tf.cast(x, tf.float32), W)

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(100):
        x_elem = np.random.permutation(x_input)
       
        
        y = list()
        for elem in x_elem:
            for i in range(len(param)):
                if np.array_equal(param[i][0], elem):
                   
                    y.append(param[i][1])
        y_elem = np.array(y)
        
        print sess.run(mul, feed_dict={x: x_elem})
       