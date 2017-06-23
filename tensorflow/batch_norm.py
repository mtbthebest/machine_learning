#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

n_inputs = 28*28
n_hidden1 = 300
n_hidden2= 100
n_outputs = 10
learning_rate = 0.01

X= tf.placeholder(dtype = tf.float32, shape = (None, n_inputs), name = 'X')
y= tf.placeholder(dtype = tf.int64, shape = (None), name = 'y')

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2/np.sqrt(n_inputs)
        init     = tf.truncated_normal((n_inputs, n_neurons) , stddev = stddev)
        W = tf.Variable(init , name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name = "biases")
        z = tf.matmul(X,W) + b
        if activation =="relu":
            return tf.nn.relu(z)
        else:
            return z
with tf.name_scope("dnn"):
    hidden1= neuron_layer(X  ,n_hidden1,"hidden1", activation = "relu")
    hidden2= neuron_layer(X  ,n_hidden2,"hidden2", activation = "relu")
    logits = neuron_layer(hidden2, n_outputs,"outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(xentropy, name = "loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()




if __name__ == '__main__':
    mnist = input_data.read_data_sets("/tmp/data")
    n_epochs=400
    batch_size = 5
    """
        Training phase
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                sess.run(training_op, feed_dict = {X:X_batch, y:y_batch})
            acc_train = accuracy.eval(feed_dict = {X:X_batch, y:y_batch})
            acc_test = accuracy.eval(feed_dict = {X:mnist.test.images, y:mnist.test.labels})
            print(epoch,"Training accuracy",acc_train, "Test accuracy", acc_test   )
        save_path = saver.save(sess,"./my_model_final.ckpt")
    """
    """
        Test phase
    """
    with tf.Session() as sess:
        saver.restore(sess,"./my_model_final.ckpt")
        X_new_scaled = mnist.test.images[0:20]
        Z = logits.eval(feed_dict = {X: X_new_scaled})
        y_pred= np.argmax(Z, axis=1)
        print ("predicted classes: " , y_pred)
        print ("actual classes: ", mnist.test.labels[0:20])