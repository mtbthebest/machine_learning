#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#Parameters
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1

n_input = 784
n_hidden_1 = 400
n_hidden_2 = 300
n_output = 10

SUM_DIR = './mnist_tens/summary'
TRAIN_DIR = './mnist_tens/train'
graph = tf.Graph()


a = np.array([[2.0,0.0],[4.0,2.0]])
b = np.array([[1.0,0.0],[1.0,0.0]])
c = tf.subtract(a,b)
d = tf.square(c)
e = tf.reduce_mean(d, axis=0)

with tf.Session() as sess:
        print sess.run([d,e])
        