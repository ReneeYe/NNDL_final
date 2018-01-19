import numpy as np

import gym
env = gym.make('Reversi8x8-v0')

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def getCanonicalForm(state,player):
    if player == 0:
        return state
    else:
        return np.array([state[1], state[0], state[2]])
    
def playAGame(agent_b=agent.RandomAgent(), agent_w=agent.RandomAgent(),env):
    agents = [agent_b, agent_w]
    player = 0
    observation = env.state
    done = False
    while True:        
        if sample['done']: # 游戏 结束
            black_score = len(np.where(env.state[0,:,:]==1)[0]) ############## 这里猪脚的程序有问题，因为可以棋盘下不满
            white_score = len(np.where(env.state[1,:,:]==1)[0])
            return black_score, white_score
        action = [65,player] 
        enables = env.possible_actions
        if len(enables) != 0:            
            action[0] = agents[player].place(observation, enables, player)#  0 表示黑棋    
        observation, reward, done, _ = env.step(action)
        player = (player + 1) % 2
        
def get_reward(board):

    black_score = len(np.where(board[0,:,:]==1)[0])
    white_score = len(np.where(board[1,:,:]==1)[0])
    if env.game_finished(board) != 0:
        return 0
    elif black_score > white_score:
        return 1
    else:
        return -1
        
    