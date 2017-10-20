#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
import numpy as np



path_dir = '/home/mtb/Downloads/'
filename= path_dir + 'port.csv'


# graph = tf.Graph()

# with graph.as_default():
#         features = tf.placeholder(tf.int32, shape=[3], name='features')
       
#         country = tf.placeholder(tf.string, name='country')
#         total = tf.reduce_sum(features, name='total')
#         printerop = tf.Print(total, [country, features, total], name='printer')

# with tf.Session(graph=graph) as sess:
    
#     sess.run( tf.global_variables_initializer())
#     with open(filename) as inf:
#         # Skip header
#         next(inf)
#         for line in inf:
#             # Read data, using python, into our features
#             country_name, code, gold, silver, bronze, total = line.strip().split(",")
#             gold = int(gold)
#             silver = int(silver)
#             bronze = int(bronze)
  
#             # Run the Print ob

#             total = sess.run(printerop, feed_dict={features: [gold, silver, bronze], country:country_name})
#             print(country_name, total)       

def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[""], [""], [0], [0], [0], [0]]
    country, code, gold, silver, bronze, total = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.pack([gold, silver, bronze])
    return features, country

filename_queue = tf.train.string_input_producer(filename)
# example, country = create_file_reader_ops(filename_queue)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print sess.run(filename_queue)
#     while True:
#         try:
#             example_data, country_name = sess.run([example, country])
#             print(example_data, country_name)
#         except tf.errors.OutOfRangeError:
#             break