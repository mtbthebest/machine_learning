#!/usr/bin/env python  
 
import os 
os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2' 
import tensorflow as tf 
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell
import numpy as np 
from collections import deque, OrderedDict 
from rulo_utils.csvreader import csvread 
from rulo_utils.csvcreater import csvcreater 
from rulo_utils.csvwriter import csvwriter 
learning_rate = 0.001 
training_epochs = 3000
batch_size = 36 
 
n_inputs = 12
n_steps = 13
n_class = 5
n_lstm_neurons = 5
model_path =os.path.abspath(os.path.join(os.path.dirname(__file__),"..")) + '/model'  
TRAIN_FILE_DIR = model_path 
SUM_DIR = model_path + '/summary' 
TRAIN_MODEL_DIR = model_path + '/train' 
# fieldnames = ['accuracy', 'iterations'] 
# csvcreater(TRAIN_MODEL_DIR '/evaluation.csv') 
graph = tf.Graph() 
 
with graph.as_default(): 
    with tf.name_scope('inputs'):  
        X = tf.placeholder(dtype=tf.float32, shape=(None,n_steps,n_inputs)) 
        y = tf.placeholder(dtype=tf.float32, shape=[None, n_class])
        p = tf.placeholder(dtype=tf.float32, shape=(None, 3,5))
         
    with tf.name_scope('rnn_dynamic'):
        # x = tf.unstack(X, n_steps, axis=1) 
        lstm_cell = BasicLSTMCell(num_units = n_lstm_neurons) 
        lstm_outputs, lstm_states = tf.nn.dynamic_rnn(lstm_cell, X, dtype =tf.float32)
        lstm_out_tansp = tf.transpose(lstm_outputs, [1,0,2])
        predicted_output = tf.gather(lstm_out_tansp, int(lstm_out_tansp.get_shape()[0])-1)
        # output_op = tf.layers.dense(lstm_outputs,n_class, activation=tf.nn.softmax) 

    with tf.name_scope('test'):
        j = tf.transpose(p,[1,0,2] )
        k = tf.gather(j,0)
with tf.Session(graph=graph) as sess: 
    init = tf.global_variables_initializer() 
    train_features= np.load('/home/mtb/Documents/data/features/' + 'train_features.npy') 
    train_labels= np.load('/home/mtb/Documents/data/features/' + 'train_labels.npy')
    test_features = np.load('/home/mtb/Documents/data/features/' + 'test_features.npy')
    test_labels = np.load( '/home/mtb/Documents/data/features/' + 'test_labels.npy')
    # print train_features.shape
    # print test_features.shape
    # print test_labels[0]
    # k = 0
    # a = train_features[200:202]
    a =np.array([[[1,2,3,4,5], [6,7,8,9,10],[11,1,12,13,15]],
                 [[1, 6, 9, 4, 5], [20,21,22,23,24], [25,2,6,27,28]]])
    print a.shape, a
    sess.run(init)   
    b = sess.run(j, {p: a}) 
    print 'b'
    print b, b.shape
    print 'c'
    c = sess.run(k, {p:a})
    print c
    
    # b ,c= sess.run([lstm_outputs, lstm_states], {X: a})
    # print b
    # d = sess.run(predicted_output, {X: a})
    # print d.shape
    # total_batch = 36
    # for k in range(36):
    #     batch_start = k * 36: 
    #     batch_stop = (k + 1) * 36
        # batch_x, batch_y = train_features[batch_start:batch_stop], train_labels[batch_start:batch_stop]
        # output= sess.run(output_op,  {X: batch_x, y: batch_y})
        # print output.shape
    #     out, sta, unstack = sess.run([lstm_outputs, lstm_states, x], { X: batch_x, y: batch_y})
    #     if :
    #             print unstack[282]
    # print len(out)

