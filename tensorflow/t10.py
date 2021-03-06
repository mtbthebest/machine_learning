#!/usr/bin/env python 

import os
os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell, static_rnn
import numpy as np
from collections import deque, OrderedDict

learning_rate = 0.001
training_epochs = 2000
batch_size = 1000

n_inputs = 8
n_outputs = 10
model_path =os.path.abspath(os.path.join(os.path.dirname(__file__),"..")) + '/model' 
TRAIN_FILE_DIR = model_path
SUM_DIR = model_path + '/summary'
TRAIN_MODEL_DIR = model_path + '/train'
# fieldnames = ['accuracy', 'iterations']
# csvcreater(TRAIN_MODEL_DIR '/evaluation.csv')
graph = tf.Graph()

with graph.as_default():
    with tf.name_scope('placeholder'):
        X0 = tf.placeholder(dtype=tf.float32, shape=[None,n_inputs])
        X1 = tf.placeholder(dtype=tf.float32, shape=[None,n_inputs])
        X = tf.placeholder(dtype=tf.float32, shape=(None,3,11))
        
    # with tf.name_scope('rnn_static'):
    #     basic_cell = BasicRNNCell(num_units = 10)
    #     outputs, states = static_rnn(basic_cell,[X0,X1], dtype=tf.float32)
    with tf.name_scope('rnn_dynamic'):
        basic_cell = BasicRNNCell(num_units = 3)
        output, state = tf.nn.dynamic_rnn(basic_cell, X, dtype =tf.float32)





with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    dirt = np.array([[1,2,3,4,5,6,7,8],[1,2,45,4,3,6,7,8],[4,2,3,4,7,6,7,0]])
    cleaning_time = np.array([[1,0,4],[1,2,5],[4,2,3]])
    sess.run(init)
    train = np.array([dirt[0], cleaning_time[0]])
    for i in range(1,3):       
            train = np.vstack([train,np.array([dirt[i], cleaning_time[i]])])
    
    for i in range(3):
        if i >0 :
            a = np.vstack([a,np.hstack(train[i])])
        else:
            a = np.hstack(train[i])    
    print a.shape
    
    c , b = sess.run([output, state], {X: np.reshape(a, (-1,3,11))})
    print c
    
