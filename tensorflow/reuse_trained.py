#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
import numpy as np


saver = tf.train.import_meta_graph("./my_model_final.ckpt.meta")

for op in tf.get_default_graph().get_operations():
    print op.name