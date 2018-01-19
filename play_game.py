# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:29:28 2018

@author: Lenovo
"""
import gym
import numpy as np

import agent

env = gym.make('Reversi8x8-v0')
    
def playGames(agent_b, agnet_w=agent.RandomAgent(), max_epochs):
    w_win =0
    b_win =0
    scores = []
    for i_episode in range(args.max_epochs):
        observation = env.reset()
        While True:               
            ################### 黑棋 B ############################### 0表示黑棋
            #  这部分 黑棋
            action = [65,0] 
            enables = env.possible_actions
            if len(enables) == 0:
                action[0] = env.board_size**2 + 1
            else:
                action[0] = agent_b.place(observation, enables, 0)#  0 表示黑棋    
            observation, reward, done, info = env.step(action)            
            ################### 白棋  W ############################### 1表示白棋
            #  这部分 白棋
            action = [65,1]
            enables = env.possible_actions
            # if nothing to do ,select pass
            if len(enables) == 0:
                action[0] = env.board_size ** 2 + 1 # pass
            else:
                action[0] = agent_w.place(observation, enables, 1)
            observation, reward, done, info = env.step(action)
            ################## GAME OVER ###########################
            if done: # 游戏 结束
                # env.render()
                black_score = len(np.where(env.state[0,:,:]==1)[0]) ############## 这里猪脚的程序有问题，因为可以棋盘下不满
                white_score = len(np.where(env.state[1,:,:]==1)[0])
                if black_score > white_score:
                    b_win += 1
                else:
                    w_win += 1
                scores.append((black_score,white_score))
                break
    return b_win, w_win, scores