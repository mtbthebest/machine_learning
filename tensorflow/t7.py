#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np

#Dataset 1

path_dir = '/home/mtb/Documents/data/train/'
filename= path_dir + 'train_2.csv'

a = np.array([[1,2,3,4],[5,6,7,8]])


def dataset1():
    dataset1 = Dataset.range(100)
    iterator = dataset1.make_initializable_iterator()
    next_elem = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)
        while True:
            try:
                b = sess.run(next_elem)
                print b
            except tf.errors.OutOfRangeError:
                break
    
    
#Dataset 2


def dataset2():
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        while True:
            try:
                b = sess.run(next_elem)
                print b
            except tf.errors.OutOfRangeError:
                break


filename_queue = tf.train.string_input_producer([filename])
line_reader = tf.TextLineReader(skip_header_lines=1)
key, value = line_reader.read(filename_queue)
# val = decode_csv(value)
# record_defaults = [[0.0], [0.0]]

# inputs, outputs = tf.decode_csv(records=value, record_defaults= record_defaults)
if __name__ == '__main__':
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
  
        for i in range(2):
            features, labels = sess.run([key,value])
            print labels[2:-3].split(',')
        coord.request_stop()
        coord.join(threads)