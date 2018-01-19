# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:29:28 2018

@author: Lenovo
"""
import gym
import numpy as np

import agent

env = gym.make('Reversi8x8-v0')

def playAGame(env, agent_b, agent_w=agent.RandomAgent(), train=False):
    agents = [agent_b, agent_w]
    sample = {}
    samples = []
    player = 0
    observation = env
    sample['done'] = False
    while True:        
        if sample['done']: # 游戏 结束
            # env.render()
            black_score = len(np.where(env.state[0,:,:]==1)[0]) ############## 这里猪脚的程序有问题，因为可以棋盘下不满
            white_score = len(np.where(env.state[1,:,:]==1)[0])
            return black_score, white_score, samples
        action = [65,player] 
        enables = env.possible_actions
        if len(enables) != 0:            
            action[0] = agents[player].place(observation, enables, player)#  0 表示黑棋    
        sample['observation'], sample['reward'], sample['done'], sample['info'] = env.step(action)
        sample['player'] = player
        samples.append(sample)
        player = (player + 1) % 2
            
def playGames(env, agent_b, agnet_w, max_epochs, train):
    w_win =0
    b_win =0
    scores = []
    trainSamples = []
    for i_episode in range(max_epochs):
        b_score, w_score, samples = playAGame(env, agent_b, agnet_w, train)
        if b_score > w_score:
            b_win += 1
        else:
            w_win += 1
        trainSamples.append(samples)
        scores.append((b_score, w_score))        
    return b_win, w_win, scores, trainSamples