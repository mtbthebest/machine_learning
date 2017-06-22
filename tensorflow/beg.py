#!/usr/bin/env python
import  os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.Variable(3, name = "x")
y = tf.Variable(4, name = "y")

f = x*x*y + y +2
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    result = f.eval()
    print result

x1 = tf.Variable(1)
if x1.graph is tf.get_default_graph():
    print 'True'
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
if x2.graph is graph:
    print 'True'
if x2.graph is not  tf.get_default_graph():
        print 'True'

w = tf.constant(3)
x = w+2
y = x+5
z = x*3
with tf.Session() as sess:
    print y.eval()
with tf.Session() as sess:
    y_val, z_val = sess.run([y,z])
    print y_val, z_val