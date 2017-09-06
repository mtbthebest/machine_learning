#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(3.0, dtype=tf.float32)
c = tf.add(a, b)

d = tf.constant([5,3,2])
e = tf.reduce_sum(d)
f= tf.reduce_prod(d)
g = tf.reduce_mean(d)
h = tf.Graph()
with tf.Session() as sess:
   print (sess.run(c))
   print sess.run(e), sess.run(f), sess.run(g)
   print sess.run([e,f,g])
  
