#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
import numpy as np
from collections import deque, OrderedDict

DIR = '/home/mtb/test.npy'

# a = np.asarray([1,2,3,4,5,6])
# np.save(DIR, a)
b =np.load(DIR)
print b.shape