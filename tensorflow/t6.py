#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size = 100
display_step = 1

n_input = 784
n_hidden_1 = 400
n_hidden_2 = 300
n_output = 10

SUM_DIR = './mnist_tens/summary'
TRAIN_DIR = './mnist_tens/train'


def layer(inputs, weights, biases):
    
        layer_1 =  tf.nn.softmax(tf.add(tf.matmul(inputs, weights['w1']) , biases['b1']))
        layer_2 =  tf.nn.softmax(tf.add(tf.matmul(layer_1, weights['w2']) , biases['b2']))
        out_layer =  tf.nn.softmax(tf.add(tf.matmul(layer_2, weights['w3']) , biases['b3']))        
        
        return out_layer


weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1],name='W1')),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],name='W2')),
    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_output],name='W3'))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),
    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),
    'b3': tf.Variable(tf.random_normal([n_output]), name='b3')
}

with tf.name_scope('IO'):
        x = tf.placeholder(tf.float32, [None, 784], name='input')
        y = tf.placeholder(tf.float32, [None, 10], name='output')
with tf.name_scope('model'):
    output = layer(x, weights, biases)

with tf.name_scope('loss'):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
        loss = tf.reduce_mean(xentropy)
        tf.summary.scalar('loss', loss)
        
with tf.name_scope('optimizer'):
       
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)

with tf.name_scope('main'):
        init = tf.global_variables_initializer()
        # saver = tf.train.Saver()

with tf.name_scope('summary'):
        summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    X_train = mnist.train.images
    X_test = mnist.test.images
    Y_train = mnist.train.labels.astype("int")
    Y_test = mnist.test.labels.astype("int")

    summary_writer = tf.summary.FileWriter(logdir=SUM_DIR)


    sess.run([init])

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            mbatch_x , mbatch_y = mnist.train.next_batch(batch_size)
            print sess.run([train_op, loss],{x: mbatch_x, y:mbatch_y})