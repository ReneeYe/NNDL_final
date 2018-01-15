# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 18:13:48 2018

@author: Think
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:21:50 2018

@author: Think
"""

import gym
import random
import numpy as np

from RL_QG_agent_TF_1 import Cbrain,Crobot
# from RL_QG_agent import RL_QG_agent
# import alpha_beta as ab

env = gym.make('Reversi8x8-v0')
env.reset()

agent_black = Crobot(flag = 0, mode = "train")
agent_white = Crobot(flag = 1, mode = "train")

# or agent合起来定义
# agent = RL_QG_agent(train = True)

# 迭代次数
MAX_EPOCHS = 20

w_win =0
b_win =0

for i_episode in range(MAX_EPOCHS):
    observation = env.reset()
    done = False
    # agent = RL_QG_agent_TF.DQN()
    # agent = RL_QG_agent()
    # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state
    for t in range(100):
        # 定义两个agent的action
        action_b = [1,2]
        action_w = [1,2]
        #或者合起来定义 action
        # action = [1,2]
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
            # action_ = agent.place(observation, enables, player) #TODO
            # action_ = agent_black.move(obs_black, enables, player)
            action_ = agent_black.place(obs_black, enables, player) #TODO
            #action_ = ab.place(observation, enables, 0)#  0 表示黑棋
        action_b[0] = action_
        action_b[1] = 0   # 黑棋 B  为 0
        
        # action = action_ , 0
        
        # observation, reward, done, info = env.step(action) # observation是黑下好之后的盘面，也是白棋面临的盘面
        observation, reward, done, info = env.step(action_b) # observation是黑下好之后的盘面，也是白棋面临的盘面
        print("BLACK:", reward)
        
        # 只是为了存一下 黑子的<SARS'> 轨迹，TODO
        # agent_black.store_transition(obs_black, action_b, reward, observation)
        
        ################### 白棋  W ############################### 1表示白棋
        # env.render()
        player = 1
        obs_white = observation
        enables = env.possible_actions
        # if nothing to do ,select pass
        if len(enables) == 0:
            action_ = env.board_size ** 2 + 1 # pass
        else:
            # action_ = agent.place(observation, enables, player)
            action_ = agent_white.place(obs_white, enables, player)
            # action_  = agent.place(observation, enables,player) # 调用自己训练的模型
            # action_ = random.choice(enables) #随机走
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
        action_w[0] = action_
        action_w[1] = 1  # 白棋 W 为 1
        # action = action_ , 1
        
        # observation, reward, done, info = env.step(action) # 这里的observation 是一回合结束后的
        observation, reward, done, info = env.step(action_w) # 这里的observation 是一回合结束后的
        print("WHITE(Me):",reward)
        
        # 存一下白子的<SARS'> 轨迹，TODO
        # agent_white.store_transition(obs_white, action_w, reward, observation)
        
        if done: # 游戏 结束
            env.render() #打印
            
            # agent.finish_episode(reward, agent.train) #TODO
            
            black_score = len(np.where(env.state[0,:,:]==1)[0])
            if black_score >32:
                print("黑棋赢了！")
                b_win += 1
            else:
                print("白棋赢了！")
                w_win += 1
            print(black_score)
            break
        
    print("黑棋：",b_win,"  白棋 ",w_win)

if agent_black.mode == "train" and agent_white.mode == "train":
    agent_black.save_model()
    agent_white.save_model()

print('黑棋胜利次数：{}\t总次数：{}'.format(b_win, MAX_EPOCHS))
print('模型胜率：{}'.format(b_win/MAX_EPOCHS))