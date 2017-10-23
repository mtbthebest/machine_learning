#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
from collections import deque, OrderedDict
#Dataset 1

path_dir = '/home/mtb/Documents/data/train/'
filename=  []
for i in [2,4,5,6,8,11,12,13,14,15,16,17,18]:
    filename.append(path_dir + 'train_' + str(i)+'.csv')
file_queue_length= [53567, 41820,94691,41931,44887,54483,57915,54964,80191,142712,75996,67732,76212]





# def convert_str_to_float(val):
#     elem_list = deque()
#     for elem in val:
#         elem_list.append(float(elem))
#     return elem_list     
    


# def convert_to_np(val_list, index):
#     data = OrderedDict()
#     # data = val_list[3][7:-1].split(',')
#     # print len(data)
#     data['inputs'] = np.asarray(convert_str_to_float(val_list[index[0]][1:-1].split(',')), dtype=np.float) 
#     data['outputs'] = np.asarray(convert_str_to_float(val_list[index[1]][7:-2].split(',')), dtype=np.float)
#     return data
   
# def decode_csv(value):

#     index = []
#     extract_data = value.split('"')
#     for elem in extract_data :
#         if len(elem)>1:
#             index.append(extract_data.index(elem))
#     [input_index, output_index] = index
#     # in_, out = convert_to_np(extract_data, index)
#     in_, out = convert_to_np(extract_data, index).values()
#     return in_, out



filename_queue = tf.train.string_input_producer(filename)
line_reader = tf.TextLineReader(skip_header_lines=1)
key, value = line_reader.read(filename_queue)

# record_defaults = [[0.0], [0.0]]

# inputs, outputs = tf.decode_csv(records=value, record_defaults= record_defaults)
if __name__ == '__main__':
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # a = list()

        for elem in file_queue_length:
            print elem
            for i in range(elem):
                features, labels = sess.run([key,value])
                # val = decode_csv(labels)
                # a.append(val[0])
                # b.append(val[1])

        # np.save('/home/mtb/test',np.array(a))
        coord.request_stop()
        coord.join(threads)
               
       