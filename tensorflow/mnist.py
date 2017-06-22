#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
import numpy as np


from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn import DNNClassifier ,RunConfig, monitors

def main(_):
    mnist = input_data.read_data_sets("/tmp/data")
    X_train = mnist.train.images
    X_test = mnist.test.images
    Y_train = mnist.train.labels.astype("int")
    Y_test = mnist.test.labels.astype("int")


    config = RunConfig(tf_random_seed = 42, save_checkpoints_secs = 10)
    feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
    validation_monitor = monitors.ValidationMonitor(x = X_test, y = Y_test, every_n_steps = 100)
    dnn_clf = DNNClassifier(hidden_units = [300, 100], n_classes = 10, feature_columns = feature_cols, config = config,
                            model_dir = "/home/mtb/Projects/machine_learning/tensorflow/mnist")

    dnn_clf.fit(X_train, Y_train, batch_size = 50, steps = 4000, monitors = [validation_monitor])
    accuracy_score = dnn_clf.evaluate(x = X_test,
                                                                                y = Y_test)["accuracy"]

    print(' accuracy_score:   {0} '.format(accuracy_score) )





if __name__ == '__main__':


    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

