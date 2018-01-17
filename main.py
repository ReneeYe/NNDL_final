# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:29:28 2018

@author: Lenovo
"""
import gym
import random
import numpy as np

import agent

env = gym.make('Reversi8x8-v0')

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
args = dotdict({
        'max_epochs': 100,
        'play_turns': 100,
        
        'checkpoint': './checkpoint/',
        'load_model': False,
        'load_folder': '',
})

if __name__ == "__main__":
    w_win =0
    b_win =0
    agent_b = agent.RandomAgent()
    agent_w = agent.GreedyAgent()
    
    for i_episode in range(args.max_epochs):
        observation = env.reset()
        # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state
        for t in range(args.play_turns):
            action = [65,0]
            # action  包含 两个整型数字，action[0]表示下棋的位置，action[1] 表示下棋的颜色（黑棋0或者白棋1）
    
            ################### 黑棋 B ############################### 0表示黑棋
            #  这部分 黑棋
            #env.render()  #  打印当前棋局
            enables = env.possible_actions
            if len(enables) == 0:
                action[0] = env.board_size**2 + 1
            else:
                action[0] = agent_b.place(observation, enables, 0)#  0 表示黑棋    
            observation, reward, done, info = env.step(action)
            # print("BLACK:", reward)
            
            ################### 白棋  W ############################### 1表示白棋
            #  这部分 白棋
            #env.render()  #  打印当前棋局
            action = [65,1]
            enables = env.possible_actions
            # if nothing to do ,select pass
            if len(enables) == 0:
                action[0] = env.board_size ** 2 + 1 # pass
            else:
                action[0] = agent_w.place(observation, enables, 1)
            observation, reward, done, info = env.step(action)
            # print("WHITE(Me):",reward)
    
            if done: # 游戏 结束
                # env.render()
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