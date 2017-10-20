#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np

#Dataset 1

path_dir = '/home/mtb/Downloads/'
filename= path_dir + 'port.csv'

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

a = np.array([[1,2,3,4],[5,6,7,8]])

with tf.Session() as sess:
    sess.run(iterator.initializer)
    while True:
        try:
            b = sess.run(next_elem)
            print b
        except tf.errors.OutOfRangeError:
            break
    

if __name__ == '__main__':
    main()
    