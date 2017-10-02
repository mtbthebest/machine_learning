#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'

import tensorflow as tf

x = tf.placeholder(tf.float32, shape = (1,0))


init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
