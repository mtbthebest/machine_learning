#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
from collections import deque, OrderedDict

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

d = tf.equal(a, b)


with tf.Session() as sess:
    c = np.asarray([1.0,2.0,3.0])
    print c
    # d = np.asarray([1.0, 2.0, 3.0])
    # e = sess.run(d, {a:c, b:d})
    # print e
    


