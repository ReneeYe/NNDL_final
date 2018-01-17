# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:21:50 2018

@author: Think
"""

import gym
import random
import numpy as np
from RL_QG_agent_TF import DQNAgent
import alpha_beta as ab

# from RL_QG_agent import RL_QG_agent
# import alpha_beta as ab

env = gym.make('Reversi8x8-v0')
env.reset()

enable_actions = np.array(range(env.action_space.n))
layer, row , col = env.observation_space.shape
model_dir = "./models_DQNvsRnd/"
# 定义我们要test的agent，比如
agent = DQNAgent(enable_actions, "reversi_expReplay_black", layer, row , col, model_dir = model_dir)

# load model
model_path = model_dir + agent.environment_name + str(200) +".ckpt"
agent.load_model(model_path = model_path)

# 迭代次数/玩100次
MAX_EPOCHS = 50

w_win =0
b_win =0

for i_episode in range(MAX_EPOCHS):
    observation = env.reset()
    # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state
    for t in range(100):
        action = [1,2]
        # action,init, 包含 两个整型数字，action[0]表示下棋的位置，action[1] 表示下棋的颜色（黑棋0或者白棋1）

        ################### 黑棋 B ############################### 0表示黑棋
        #  这部分 黑棋 是 用 alpha-beta搜索
        # env.render()  #  打印当前棋局
        player = 0
        obs_black = observation # 记录当前状黑棋面临的盘面，也就是每一个回合一开始的盘面
        enables = env.possible_actions
        if len(enables) == 0:
            action_ = env.board_size**2 + 1
        else:
            # action_ = random.choice(enables)
            action_ = ab.place(observation, enables, 0)  # 0 表示黑棋
        action[0] = action_

        action[1] = 0  # 黑棋 B  为 0
        observation, reward, done, info = env.step(action)
        # print("BLACK:", reward)

        ################### 白棋  W ############################### 1表示白棋
        # env.render()
        player = 1
        obs_white = observation
        enables = env.possible_actions
        # if nothing to do ,select pass
        if len(enables) == 0:
            action_ = env.board_size ** 2 + 1 # pass
        else:
            _, action_ = agent.select_enable_action(observation, enables)
            # action_ = agent_white.place(obs_white, enables, player)
            # action_  = agent.place(observation, enables,player) # 调用自己训练的模型
            """
            zidian = dict()# 这部分是方便人工输入，根据提示的坐标，输入对应的位置编号
            for e in enables:
                xx = e//8
                yy = e - xx*8
                zidian[(xx+1,yy+1)] = e
            print(zidian)
            inp = input('>>>  ')
            action_ = int(inp) ###########
            """

        action[0] = action_
        action[1] = 1
        
        observation, reward, done, info = env.step(action) # 这里的observation 是一回合结束后的
        # observation, reward, done, info = env.step(action_w) # 这里的observation 是一回合结束后的
        # print("WHITE(Me):",reward)

        
        if done: # 游戏 结束
            env.render() #打印
            
            black_score = len(np.where(env.state[0,:,:]==1)[0])
            white_score = len(np.where(env.state[1, :, :] == 1)[0])
            if black_score >32:
                print("EPOCH: {:03d}/{:03d}: 黑棋赢了！".format(i_episode, MAX_EPOCHS))
                b_win += 1
            else:
                print("EPOCH: {:03d}/{:03d}: 白棋赢了！".format(i_episode, MAX_EPOCHS))
                w_win += 1
                print("盘面棋子数: black:", black_score, "  white:", white_score)
            break
        
    print("黑棋(minimax)：",b_win,"  白棋(DQN) ",w_win)


print('白棋胜利次数：{}\t总次数：{}'.format(w_win, MAX_EPOCHS))
print('DQN vs. minimax 模型胜率：{}'.format(w_win/MAX_EPOCHS))