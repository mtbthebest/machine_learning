#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
import numpy as np
a = tf.add(2,5)
b = tf.multiply(a,3)

e = tf.placeholder(dtype=tf.int32, shape=[2], name='array')
f = tf.reduce_sum(e)
with tf.Session() as sess:
    c = sess.run(fetches=b)
    d = sess.run(fetches=b, feed_dict={a:15})
    print [c,d]
    g= sess.run(f, feed_dict={e: np.array([5,3])})
    print g