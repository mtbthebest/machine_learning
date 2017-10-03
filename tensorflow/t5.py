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



# def layer(input_, weights_name, bias_name, weights_shape, bias_shape):
        
#         with tf.variable_scope('variables') as scope:          
        
#             W = tf.get_variable(weights_name, weights_shape)
            
#             b = tf.get_variable(bias_name, bias_shape)
            
#             Out =  tf.nn.softmax(tf.add(tf.matmul(input_, W) , b))
            
#             return Out

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# hidden_1 = layer(x, 'w1','b1',(784, neuron_1), (neuron_1))
    
# hidden_2 = layer(hidden_1, 'w2', 'b2',(neuron_1, neuron_2), (neuron_2))

# output = layer(hidden_2, 'w3', 'b3',(neuron_2, neuron_3), (neuron_3))
W = tf.get_variable('w',[784,10])

b = tf.get_variable('b', [10])

output = tf.nn.softmax(tf.add(tf.matmul(x, W), b))
xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
loss = tf.reduce_mean(xentropy)
        
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

tf.summary.scalar('loss', loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

summary_op = tf.summary.merge_all()

with tf.Session(graph=tf.get_default_graph()) as sess:
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    X_train = mnist.train.images
    X_test = mnist.test.images
    Y_train = mnist.train.labels.astype("int")
    Y_test = mnist.test.labels.astype("int")
    
    summary_writer = tf.summary.FileWriter(logdir=SUM_DIR, graph=tf.get_default_graph())

    
    sess.run([init])
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            mbatch_x , mbatch_y = mnist.train.next_batch(batch_size)
            # sess.run(hidden_2,{ x:mbatch_x})
            # print 'y' + str(mbatch_y.shape)
            # a = sess.run(output,{ x:mbatch_x})
            # print 'output' + str(a.shape)

            # print tf.global_variables()
            # a = sess.run(loss,{x: mbatch_x, y:mbatch_y})
            # print a
            train, cost_function, summary = sess.run([train_op,loss,summary_op], feed_dict={x: mbatch_x, y:mbatch_y})
            print cost_function
            # print W
            # train_operation, accuracy_run, minibatch_cost = sess.run(
            #     fetches=[train_op, accuracy, loss], feed_dict={x: mbatch_x, y: mbatch_y})
           
           
            # print accuracy_run
            # avg_cost += minibatch_cost / total_batch


# with tf.variable_scope('variables'):
#     W = tf.get_variable('W',dtype=tf.float32,initializer=0.0)
#     # W = tf.Variable(0.0,trainable=False)

# with tf.name_scope('test'):
#     a = W + (1.0)
# with tf.name_scope('as'):
#     b = tf.multiply(W, tf.constant(2.0))
# a = tf.Variable(initial_value=[[2,4,8],[1,4,3]], trainable=False)
# b = tf.Variable(initial_value=[[1, 1, 9], [1, 4, 7]], trainable=False)

# c = tf.equal(tf.argmax(a, 1), tf.argmax(b,1))
# d= tf.cast(c, tf.float32)
# # c = tf.square((a-b))

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(d)
    # for i in range(10):
    #     print sess.run(a)
    
    # print sess.run(b)

    

            
        
        
            
