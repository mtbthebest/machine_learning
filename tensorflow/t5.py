#!/usr/bin/env python
import os

os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#Parameters
learning_rate = 0.01
training_epochs = 1000
batch_size =100
display_step = 1

neuron_1 = 500
neuron_2 = 400
neuron_3 = 10

SUM_DIR = './mnist_tens/summary'
TRAIN_DIR = './mnist_tens/model.ckpt'
graph = tf.Graph()



tf_graph = tf.Graph()

with tf_graph.as_default():

    def layer(input_, weights_shape, bias_shape, weight_name, bias_name):
        
        with tf.variable_scope('variables'):
            W = tf.get_variable(name= weight_name, initializer=tf.truncated_normal(shape=weights_shape, stddev= 2/np.sqrt(784)))
            b = tf.get_variable(name= bias_name, initializer=tf.zeros(bias_shape))
        
        return tf.nn.relu(tf.add(tf.matmul(input_, W), b))
    
    with tf.name_scope('placeholder'):
        x = tf.placeholder(tf.float32, [None, 784], name= 'x')
        y = tf.placeholder(tf.float32, [None, 10], name= 'y')

    with tf.name_scope('layers'):
        hidden_1 = layer(x, [784, neuron_1], [neuron_1],'w1','b1')    
        hidden_2= layer(hidden_1,[neuron_1, neuron_2], [neuron_2],'w2','b2')
        output= layer(hidden_2,[neuron_2, neuron_3], [neuron_3],'w3','b3') 

    with tf.name_scope('loss'):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
        loss = tf.reduce_mean(xentropy)
        tf.summary.scalar('loss', loss)

    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0, trainable=False,name='global_step')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    with tf.name_scope('initializer'):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
    
        
with tf.Session(graph=tf_graph) as sess:
   
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    X_train = mnist.train.images
    X_test = mnist.test.images
    Y_train = mnist.train.labels.astype("int")
    Y_test = mnist.test.labels.astype("int")    
   
    
    total_iterations = 0
    # saver = tf.train.import_meta_graph('./mnist_tens/train.meta')
    # for op in tf.get_default_graph().get_operations():
    #     print op.name

    if os.path.isfile('./mnist_tens/checkpoint'):
        print 'Restoring'
        saver.restore(sess,TRAIN_DIR)
    summary_writer = tf.summary.FileWriter(logdir=SUM_DIR, graph=tf_graph)    
    
       
    sess.run([init])

    for epoch in range(training_epochs):
        
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            mbatch_x , mbatch_y = mnist.train.next_batch(batch_size)
           
            train, cost_function, acc , summary= sess.run([train_op,loss,accuracy,summary_op], feed_dict={x: mbatch_x, y:mbatch_y})
            total_iterations +=1
            
            step = tf.train.global_step(sess, global_step)
            summary_writer.add_summary(summary, global_step= step)
            
            print step

            print ('Iterations %d, loss: %.6f'%(total_iterations, cost_function))
        
        print('Saving epoch {0}'.format(epoch))
        saver.save(sess,TRAIN_DIR)
     
          
            # print W
            # train_operation, accuracy_run, minibatch_cost = sess.run(
            #     fetches=[train_op, accuracy, loss], feed_dict={x: mbatch_x, y: mbatch_y})
           
           
            # print accuracy_run
            # avg_cost += minibatch_cost / total_batch


# with tf.variable_scope('variables'):
#     W = tf.get_variable('W',dtype=tf.float32,initializer=0.0)
#     # W = tf.Variable(0.0,trainable=False)

# with tf.name_scope('test'):
#     a = W + (1.0)
# with tf.name_scope('as'):
#     b = tf.multiply(W, tf.constant(2.0))
# a = tf.Variable(initial_value=[[2,4,8],[1,4,3]], trainable=False)
# b = tf.Variable(initial_value=[[1, 1, 9], [1, 4, 7]], trainable=False)

# c = tf.equal(tf.argmax(a, 1), tf.argmax(b,1))
# d= tf.cast(c, tf.float32)
# # c = tf.square((a-b))

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print sess.run(d)
    # for i in range(10):
    #     print sess.run(a)
    
    # print sess.run(b)

    

            
        
        
            
