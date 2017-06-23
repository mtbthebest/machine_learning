#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial

n_inputs = 28*28
n_hidden1 = 300
n_hidden2= 100
n_outputs = 10
learning_rate = 0.01
batch_norm_momentum = 0.9


X= tf.placeholder(dtype = tf.float32, shape = (None, n_inputs), name = 'X')
y = tf.placeholder(tf.int64, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape =(), name="training")


with tf.name_scope("dnn"):
    he_init = tf.contrib.layers.variance_scaling_initializer()

    batch_norm_layer = partial(tf.layers.batch_normalization,training=training, momentum=batch_norm_momentum)
    dense_layer = partial(tf.layers.dense, kernel_initializer = he_init)
   
    hidden1= dense_layer(X,n_hidden1,name = "hidden1")
    bn1 = tf.nn.elu(batch_norm_layer(hidden1))
    
    hidden2 = dense_layer(bn1, n_hidden2, name="hidden2")
    bn2 = tf.nn.elu(batch_norm_layer(hidden2))

    logits_before_bn = dense_layer(bn2, n_outputs, name="outputs")
    logits = batch_norm_layer(logits_before_bn)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
init = tf.global_variables_initializer()
saver = tf.train.Saver()




if __name__ == '__main__':
    try:
        mnist = input_data.read_data_sets("/tmp/data")
        n_epochs = 20
        batch_size = 100
            
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
    
            saver.restore(sess,"./my_model_final.ckpt")
            X_new_scaled = mnist.test.images[0:20]
            Z = logits.eval(feed_dict = {X: X_new_scaled})
            y_pred= np.argmax(Z, axis=1)
            print ("predicted classes: " , y_pred)
            print ("actual classes: ", mnist.test.labels[0:20])
    except:
        pass