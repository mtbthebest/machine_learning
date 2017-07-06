#!/usr/bin/env python
import os
import sys
import gym
os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer, conv2d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy.random as rnd

tf.reset_default_graph()
env = gym.make("MsPacman-v0")



def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=40):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval)



def plot_environment(env, figsize=(5,4)):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    plt.show()



frames = []

n_max_steps = 1000
n_change_steps = 10

if __name__ == '__main__':
    obs = env.reset()
    total_reward = 0
    print env.action_space
    for step in range(n_max_steps):
        img = env.render(mode="rgb_array")
        frames.append(img)
        if step % n_change_steps == 0:
            action = env.action_space.sample() # play randomly
        obs, reward, done, info = env.step(action)
        total_reward +=reward
        if done:
            print ('done')           
            break
    print('step ', step , ':' ,total_reward)
    video = plot_animation(frames)
    plt.show()
