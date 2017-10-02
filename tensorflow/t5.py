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

neuron_1 = 400
neuron_2 = 300
neuron_3 = 10

SUM_DIR = './mnist_tens/summary'
TRAIN_DIR = './mnist_tens/train'
graph = tf.Graph()

with graph.as_default():

    def layer(input_, weight_name, bias_name, weights_shape, bias_shape):
        
        with tf.variable_scope('variables'):
        
            W = tf.get_variable(weight_name, initializer=tf.random_uniform(weights_shape,minval=0, maxval=1))
            
            b = tf.get_variable(bias_name, initializer=tf.zeros(bias_shape))
            
            Out =  tf.nn.softmax(tf.add(tf.matmul(input_, W) , b))
            
            return Out

    with tf.name_scope('IO'):
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope('layers'):
                  
            hidden_1 = layer(x, 'w1','b1',(784, neuron_1), (neuron_1))
            
            hidden_2 = layer(hidden_1, 'w2', 'b2',(neuron_1, neuron_2), (neuron_2))
           
            output = layer(hidden_2, 'w3', 'b3',(neuron_2, neuron_3), (neuron_3))
           
       
    with tf.name_scope('loss'):
        diff = tf.square((y-output))
        loss = tf.reduce_mean(diff)
        tf.summary.scalar('loss', loss)
        
    with tf.name_scope('optimizer'):
       
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
    
    # with tf.name_scope('evaluation'):
    #     correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y,1))
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
               


    with tf.name_scope('main'):
        init = tf.global_variables_initializer()
        # saver = tf.train.Saver()


    with tf.name_scope('summary'):
        summary_op = tf.summary.merge_all()

with tf.Session(graph=graph) as sess:
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    X_train = mnist.train.images
    X_test = mnist.test.images
    Y_train = mnist.train.labels.astype("int")
    Y_test = mnist.test.labels.astype("int")
    
    summary_writer = tf.summary.FileWriter(logdir=SUM_DIR, graph=graph)

    
    sess.run([init])
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            mbatch_x , mbatch_y = mnist.train.next_batch(batch_size)
            # sess.run(hidden_2,{ x:mbatch_x})

            # print tf.global_variables()
            train, cost_function, summary = sess.run([train_op,loss,summary_op], feed_dict={x: mbatch_x, y:mbatch_y})
            print cost_function
            # train_operation, accuracy_run, minibatch_cost = sess.run(
            #     fetches=[train_op, accuracy, loss], feed_dict={x: mbatch_x, y: mbatch_y})
           
           
            # print accuracy_run
            # avg_cost += minibatch_cost / total_batch

            
        
        
            
