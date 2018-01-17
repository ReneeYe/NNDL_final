# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:46:58 2018

@author: Lenovo
"""
import random
import alpha_beta

directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

class Agent():
    def __init__(self, train=False):
        self.train = train
    
    def place(self, state, enables, player):
        """
        the agent will receive a game state and a player, alonge with all legal actions
        the agent should return its chozen action
        """
        pass
    
class GreedyAgent(Agent):
    
    def num_to_cord(self, action):
        return((int(action/8),action % 8))
    def cord_to_num(self, cord):
        return(cord[0]*8+cord[1])  
    
    def positionLegal(self, cord):
        if (cord[0]<0 or cord[0]>=8 or cord[1]<0 or cord[1]>=8):
            return False
        else:
            return True
        
    def countReverse(self, board, action, player):  # 这里就是 进行下棋操作。并更新棋盘
        cord = self.num_to_cord(action)
        reverse_sum = 0
        for direction in directions:
            count = 0
            current_position = (cord[0]+direction[0],cord[1]+direction[1])
            while self.positionLegal(current_position):
                if board[(player+1) % 2][current_position[0]][current_position[1]]:                
                    count += 1
                    current_position = (current_position[0]+direction[0],current_position[1]+direction[1])
                else:
                    break
            if (self.positionLegal(current_position)) and (board[player][current_position[0]][current_position[1]]):
                reverse_sum = reverse_sum + count
        return reverse_sum
    
    def place(self, state, enables, player):
        max_reverse_num = 0
        max_reverse_action = 65
        for i in range(len(enables)):
            reverse_num = self.countReverse(state, enables[i], player)
            if reverse_num > max_reverse_num:
                max_reverse_num = reverse_num
                max_reverse_action = enables[i]
        return max_reverse_action
        

class RandomAgent(Agent):
    def place(self, state, enables, player):
        return random.choice(enables)    

class AlphaBetaAgent(Agent):
    def place(self, state, enables, player):
        return alpha_beta.place(state, enables, player)