#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size = 100
display_step = 1

neuron_1 = 500
neuron_2 = 400
neuron_3 = 10

SUM_DIR = './mnist_fold/summary'
TRAIN_DIR = '/home/mtb/Projects/machine_learning/tensorflow/mnist_fold/model/model.ckpt'
graph = tf.Graph()

x = tf.placeholder(tf.float32, [None, 784], name='x')
def layer(scope_name):
    # x = tf.placeholder(tf.float32, [None, 784], name='x')
    
    H1 = tf.layers.dense(x, neuron_1, activation=tf.nn.relu,
                        kernel_initializer=variance_scaling_initializer())
    H2 = tf.layers.dense(H1, neuron_2, activation=tf.nn.relu,
                        kernel_initializer=variance_scaling_initializer())
    O = tf.layers.dense(H2, neuron_3, activation=tf.nn.tanh,
                        kernel_initializer=variance_scaling_initializer())
    W = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)
    return O

def network(scope_name):
    try:
        with tf.variable_scope(scope_name, reuse=True) as scope:
            network =layer(scope_name)
    except:
        with tf.variable_scope(scope_name,reuse=False) as scope:
            network = layer(scope_name)  
    return network

y = tf.placeholder(tf.float32, [None, 10], name='y')

with tf.name_scope('loss'):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=network('test'), labels=y)
    loss = tf.reduce_mean(xentropy)
    tf.summary.scalar('loss', loss)

with tf.name_scope('optimizer'):
    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step)

# with tf.name_scope('accuracy'):
#     correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('initializer'):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()


with tf.Session() as sess:

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    X_train = mnist.train.images
    X_test = mnist.test.images
    Y_train = mnist.train.labels.astype("int")
    Y_test = mnist.test.labels.astype("int")

    total_iterations = 0
    
    sess.run([init])
    summary_writer = tf.summary.FileWriter(logdir=SUM_DIR)

    if os.path.isfile('/home/mtb/Projects/machine_learning/tensorflow/mnist_fold/model/checkpoint'):
        print 'Restoring'
        saver.restore(sess,TRAIN_DIR)

    for epoch in range(training_epochs):

        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)
            train , cost_function = sess.run([train_op,loss], feed_dict={x: mbatch_x, y: mbatch_y})
            print cost_function
        #     train, cost_function, acc, summary = sess.run(
        #         [train_op, loss, accuracy, summary_op], feed_dict={x: mbatch_x, y: mbatch_y})
        #     total_iterations += 1

        #     step = tf.train.global_step(sess, global_step)
        saver.save(
            sess, TRAIN_DIR)
        #     summary_writer.add_summary(summary, global_step=step)

        #     print step

        #     print ('Iterations %d, loss: %.6f' %
        #            (total_iterations, cost_function))

        # print('Saving epoch {0}'.format(epoch))
