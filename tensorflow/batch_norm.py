#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers.batch_norm() import batch_norm


n_inputs = 28*28
n_hidden1 = 300
n_hidden2= 100
n_outputs = 10
learning_rate = 0.01

X= tf.placeholder(dtype = tf.float32, shape = (None, n_inputs), name = 'X')
training = tf.placeholder_with_default(False, shape =(), name="training")

hidden1= tf.layers.dense(X,n_hidden1,name = "hidden1")
bn1 = tf.layers.batch_normalization(hidden1, trainable=training, momentum=0.9)
bn1_act = tf.nn.elu(bn1)

hidden2= tf.layers.dense(bn1_act,n_hidden2,name = "hidden2")
bn2 = tf.layers.batch_normalization(hidden2, trainable=training, momentum=0.9)
bn2_act = tf.nn.elu(bn2)

logits_before_bn = tf.layers.dense(bn1_act,n_outputs,name = "outputs")
logits = tf.layers.batch_normalization(logits_before_bn, training=training, momentum=0.9)

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