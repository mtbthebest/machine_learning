#!/usr/bin/env python
import os
import gym
import sys
import time
os.environ['TF_CPP_MIN_LOG_LEVLEL'] = '2'
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np



try:
    from pyglet.gl import gl_info
    openai_cart_pole_rendering = True   # no problem, let's use OpenAI gym's rendering function
except Exception:
    openai_cart_pole_rendering = False  # probably no X server available, let's use our own rendering function

def render_cart_pole(env, obs):
    if openai_cart_pole_rendering:
        # use OpenAI gym's rendering function
        return env.render(mode="rgb_array")
    else:
        # rendering for the cart pole environment (in case OpenAI gym can't do it)
        img_w = 600
        img_h = 400
        cart_w = img_w // 12
        cart_h = img_h // 15
        pole_len = img_h // 3.5
        pole_w = img_w // 80 + 1
        x_width = 2
        max_ang = 0.2
        bg_col = (255, 255, 255)
        cart_col = 0x000000 # Blue Green Red
        pole_col = 0x669acc # Blue Green Red

        pos, vel, ang, ang_vel = obs
        img = Image.new('RGB', (img_w, img_h), bg_col)
        draw = ImageDraw.Draw(img)
        cart_x = pos * img_w // x_width + img_w // x_width
        cart_y = img_h * 95 // 100
        top_pole_x = cart_x + pole_len * np.sin(ang)
        top_pole_y = cart_y - cart_h // 2 - pole_len * np.cos(ang)
        draw.line((0, cart_y, img_w, cart_y), fill=0)
        draw.rectangle((cart_x - cart_w // 2, cart_y - cart_h // 2, cart_x + cart_w // 2, cart_y + cart_h // 2), fill=cart_col) # draw cart
        draw.line((cart_x, cart_y - cart_h // 2, top_pole_x, top_pole_y), fill=pole_col, width=pole_w) # draw pole
        return np.array(img)


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

if __name__ == '__main__':
    env =gym.make("CartPole-v0")
    env.reset()
    env.render()
    totals = []
    for episodes in range(500):
        episodes_rewards =0
        obs = env.reset() 
        for steps in range(1000):
            action = basic_policy(obs)
            obs, reward , done, info = env.step(action)
            episodes_rewards += reward
            plt.close()  # or else nbagg sometimes plots in the previous cell
            img = render_cart_pole(env, obs)
            plt.imshow(img)
            plt.axis("off")
            
            if done:
                 break
    
        
        totals.append(episodes_rewards)

print np.mean(totals), np.std(totals), np.min(totals), np.max(totals)