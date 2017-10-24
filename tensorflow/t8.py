#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
from collections import deque, OrderedDict

DIR = '/home/mtb/Documents/data/train/inputs.npy'

filename = DIR
input_=np.load(DIR)
# input_ = np.array([[1,2,3.0],[4.0,5.0,6.0], [7.0,8.0,9.0],[10.0,11.0,12.0]])
# output_ = np.array([[1,2,3.0],[4.0,5.0,6.0]])

input_placeholder = tf.placeholder(tf.float32, shape =None)
# output_placeholder = tf.placeholder(tf.float32, shape = output_.shape)
a =Dataset.from_tensors(input_placeholder)
# b = Dataset.from_tensors(output_placeholder)
# c = Dataset.zip((a,b))
# batch_ = a.batch(100)
iterator=a.make_initializable_iterator()
next_elem = iterator.get_next()



# # b = Dataset.from_tensor_slices(input_placeholder)
# # tf.Session().run(tf.global_variables_initializer(),)
# # b = Dataset.from_tensor_slices(tf.Session().run(v,{input_placeholder: input_}))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
   
    # print a.shape
    # print sess.run(next_elem, {input_placeholder:input_})
    input_dict = OrderedDict() 

        
        
    for k in range(7):
        start = 100000*k
        stop = ( k+1) * 100000
        input_dict[str(k)] = deque()
        for i in range(start, stop):            
            if input_[i].shape[0]==720:
                input_dict[str(k)].append(input_[i])
       
    
    
    sess.run(iterator.initializer, {input_placeholder:input_dict[str(1)]}) 

    a =  sess.run(next_elem, {input_placeholder:input_dict[str(1)]})
    print a
    

    

