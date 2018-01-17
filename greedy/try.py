# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 10:51:31 2018

@author: Lenovo
"""
import gym
import random
import numpy as np

import greedy

env = gym.make('Reversi8x8-v0')
action = [0,1]
env.render()  #  打印当前棋局
enables = list(set(env.possible_actions))
if len(enables) == 0:
    action_ = env.board_size**2 + 1
else:
    action_ = random.choice(enables)
action[0] = action_
action[1] = 0   # 黑棋 为 0
observation, reward, done, info = env.step(action)
print (observation[0])

env.render()
enables = list(set(env.possible_actions))
# if nothing to do ,select pass
if len(enables) == 0:
    action_ = env.board_size ** 2 + 1 # pass
else:
    # action_ = random.choice(enables)
    player = 1
    action_  = greedy.place(observation, enables,player) # 调用自己训练的模型

action[0] = action_
action[1] = player  # 白棋 为 1
observation, reward, done, info = env.step(action)
