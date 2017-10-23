#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
from collections import deque, OrderedDict
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


def convert_str_to_float(val):
    elem_list = deque()
    for elem in val:
        elem_list.append(float(elem))
    return elem_list     
    


def convert_to_np(val_list, index):
    data = OrderedDict()
    # data = val_list[3][7:-1].split(',')
    # print len(data)
    data['inputs'] = np.asarray(convert_str_to_float(val_list[index[0]][1:-1].split(',')), dtype=np.float) 
    data['outputs'] = np.asarray(convert_str_to_float(val_list[index[1]][7:-2].split(',')), dtype=np.float)
    return data
   
def decode_csv(value):

    index = []
    extract_data = value.split('"')
    for elem in extract_data :
        if len(elem)>1:
            index.append(extract_data.index(elem))
    [input_index, output_index] = index
    # in_, out = convert_to_np(extract_data, index)
    in_, out = convert_to_np(extract_data, index).values()
    return in_, out



filename_queue = tf.train.string_input_producer([filename], capacity=1000)
line_reader = tf.TextLineReader(skip_header_lines=1)
key, value = line_reader.read(filename_queue)

# record_defaults = [[0.0], [0.0]]

# inputs, outputs = tf.decode_csv(records=value, record_defaults= record_defaults)
if __name__ == '__main__':
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
  
        while True:
            try:
                features, labels = sess.run([key,value])
                val = decode_csv(labels)
                print val[0]
            except tf.errors.OutOfRangeError:
                break
        coord.request_stop()
        coord.join(threads)
