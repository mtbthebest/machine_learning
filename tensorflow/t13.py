#!/usr/bin/env python
import os
import tensorflow as tf
import numpy as np


y_labels = tf.placeholder(dtype=tf.float32, shape=[None, 2], name= 'labels')
y_predict = tf.placeholder(dtype=tf.float32, shape=[None, 2], name= 'predict')

actual_negative = tf.equal(tf.argmax(y_labels, axis=1), 1)
actual_positive = tf.equal(tf.argmax(y_labels, axis=1), 0)
predict_positive = tf.equal(tf.argmax(y_predict, axis=1), 0)
predict_negative = tf.equal(tf.argmax(y_predict, axis=1), 1)

true_negative = tf.reduce_sum(tf.cast(tf.logical_and(actual_negative, predict_negative), tf.int32))
true_positive =tf.reduce_sum(tf.cast(tf.logical_and(actual_positive, predict_positive), tf.int32))
false_negative=tf.reduce_sum(tf.cast(tf.logical_and(actual_positive, predict_negative), tf.int32))
false_positive = tf.reduce_sum(tf.cast(tf.logical_and(actual_negative, predict_positive), tf.int32))
with tf.Session() as sess:
    labels = np.array([
            [0.0,1.0],
            [0.0,1.0],
            [0.0,1.0],
            [1.0,0.0],
            [1.0,0.0],
            [0.0,1.0],
            [1.0,0.0],
            [1.0,0.0],
            [0.0,1.0],
            [1.0,0.0],
            [0.0,1.0]
    ])

    predicted = np.array([
            [0.0,1.0],
            [0.0,1.0],
            [0.0,1.0],
            [0.0,1.0],
            [0.0,1.0],
            [0.0,1.0],
            [1.0,0.0],
            [1.0,0.0],
            [0.0,1.0],
            [1.0,0.0],
            [1.0,0.0]
    ])
    #   Class 1 rank 1
    tn,tp, fn , fp = sess.run([true_negative,true_positive, false_negative, false_positive] , {y_labels: labels, y_predict: predicted})
    confusion_matrix = np.array([[tn,fp], [fn, tp]])
    print confusion_matrix

    precision = float(confusion_matrix[1][1]) / float(np.sum(confusion_matrix[:,1]))
    recall =  float(confusion_matrix[1][1]) / float(np.sum(confusion_matrix[1]))
    print precision, recall
  
