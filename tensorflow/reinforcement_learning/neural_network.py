#!/usr/bin/env python
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
from tensorflow.contrib.layers import variance_scaling_initializer
import tensorflow as tf

tf.reset_default_graph()

n_inputs = 4
n_hidden = 4
n_outputs = 1

learning_rate = 0.01

initializer = variance_scaling_initializer()
X = tf.placeholder(tf.float32, shape=[None,n_inputs])
y = tf.placeholder(tf.float32, shape=[None,n_outputs])

hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits)
p_left_and_right = tf.concat(axis=1, values =[outputs, 1-outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples = 1)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer= tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

if __name__ == '__main__':
    n_environments = 10
    n_iterations = 1000
    envs = [gym.make("CartPole-v0") for _ in range(n_environments)
    observations = [env.reset() for env in envs]
    
    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            target_probas = np.array([([1.] if obs[2] < 0 else [0.]) for obs in observations]) # if angle<0 we want proba(left)=1., or else proba(left)=0.
            action_val, _ = sess.run([action, training_op], feed_dict={X: np.array(observations), y: target_probas})
            for env_index, env in enumerate(envs):
                obs, reward, done, info = env.step(action_val[env_index][0])
                observations[env_index] = obs if not done else env.reset()
        saver.save(sess, "./my_policy_net_basic.ckpt")

    for env in envs:
        env.close()