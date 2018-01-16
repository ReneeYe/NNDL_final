# Deep Reinforcement Learning for Othello (Reversi)

Final Project of DATA130011.01 Neural Network and Deep Learning
School of Data Sciences, Fudan University 

**Rong Ye**

**Yuqing Xie** 14307130361@fudan.edu.cn, 
Mathematics and Applied Mathematics, School of Mathematical Sciences  

## Abstract

## Introduction

### The Othello Game

【rule】[mainly from paper 1(**to be modified**)] Othello is a two-player game on a 8 * 8 board. There are 64 identical pieces which are white on one side and black on the other. The game begins with each player having two pieces placed diagonally in the center of the board **(Figure 1a)**. The black player moves first. A move is legal if the newly placed piece is adjacent to an opponent’s piece and causes one or more of the opponent’s pieces to become enclosed from both sides on a horizontal, vertical or diagonal line. The enclosed pieces are then flipped. The black player's first move can be any one of four shaded locations, in **(Figure 1b)**. Players alternate placing pieces on the board. If and only if a player does not have any legal move the player passes. The game ends when neither player has a legal move(of course when the board is filled with pieces), and the player with more pieces on the board wins. If both players have the same number of pieces, the game ends in a draw.

【features & difficulity】Othello is a <u>perfect information, zero-sum</u>, two-player strategy game. Despite simple rules, the game is far from trivial. The number of <u>legal positions</u> is approximately 10^28 and the <u>game tree</u> has in the order of 10^58 nodes [Victor L Allis. Searching for solutions in games and artificial intelligence. PhD thesis, University of Limburg, Maastricht, The Netherlands,1994], which precludes any exhaustive search method. Othello is also characterized by a <u>high temporal volatility</u>: a high number of pieces can be flipped in a single move, dramatically changing the board state. 

### Related work

[1] introduced CNN and using professional game records for training

[2] mainly compares different reinforcement learning methords' performance

[3] adversarial advantage for task-compeletion dialogue policy learning

[4] The best known Othello playing program is LOGISTELLO.

## Reinforcement learning model

Basic reinforcement learning model:

In reinforcement learning, the learner is a decision making agent that takes actions in an environment and receives a reward (or penalty) for its actions in trying to solve a problem. After a set of trial-and-error runs it should learn the best policy, which is the sequence of actions that maximize the total reward. 

We define our problem as a Markov decision process.
(1) A finite set of states s \belongs to S; In othello, s means a certain broad state and S contains all possible states of the game, approximately 10^28 states.
(2) A finite set of actions a \belongs to A; In othello, a means a legal position to place piece.
(3) The transition function T(s, a, s') is simply to place a piece chosen by action a after state s.
(4) A reward function R(s, a), providing the reward the agent will receive for executing action a in state s, where r_t denotes the reward obtained at time t; In Othello, a reward is received only when the whole game comes to an end. Also in our project, official judgement does not consider how many piece is winned, but only care whether black player win the game. We will try this reward function and also consider other choice to improve the agent's performance.
(5) A discount factor 0<= \gamma <=1 which discounts later rewards compared to immediate rewards.

In a reinforcement learning model, we also have to define value functions: 
We want our agent to learn an optimal policy for mapping states to actions. The actions to be taken in any states s : a = \pi(s). The value of a policy \pi, V^(\pi)(s), is the expected cummulative reward that will be received when the agent follows the policy starting at state s. It is defined as: **\VALUE FUNCTION**, we also use Q-value for there is no use in adding s' in to 自变量

Learning algorithms

TD- network
1) Observe the current state st
2) For all afterstates s0 t reachable from st use NN to compute V (s0 t)
3) Select an action leading to afterstate sa t using a policy π
4) According to (10) compute the target value of the previous afterstate V new(sa t−1)
5) Use NN to compute the current value of the previous afterstate V (sa t−1)
6) Adjust the NN by backpropating the error V new(sa t−1)− V (sa t−1)
7) sa t−1 sa t
8) Execute action resulting in afterstate sa t

Q-learing newtwork
1) Observe the current state st
2) For all possible actions a0 t in st use NN to compute Q ^(st; a0 t)
3) Select an action at using a policy π
4) According to (8) compute the target value of the previous state-action pair Q ^new(st−1; at−1)1) Observe the current state st
5) Use NN to compute the current estimate of the value of the previous state-action pair Q ^(st−1; at−1)
6) Adjust the NN by backpropating the error Q ^new(st−1; at−1) − Q ^(st−1; at−1)
7) st−1 st, at−1 at
8) Execute action at

Here are some points that we should focus on:
1) According to the huge amount of states, we have to learn the policy using statistic methods and network structure may affect learning proficiency greatly.
2) The reword function is also critical for learning ability
3) The algorithms' performance also vary when the opponent change. We have fixed policy opponents, experts playing records, best players in the world and we also tried self-play opponent inspired by Alpha-Go.

## Experiement

### Dataset(if used the professional database)

### Learning& test – results

### Compare& analysis

## Reference

[1] Learning toPlay Othello with Deep Neural Networks（CNN的那篇）
[2] ReinforcementLearning in the Game of Othello: Learning Against a Fixed Opponent and Learningfrom Self-Play（主要参考程序章结构）
[3] ADVERSARIALADVANTAGE ACTOR-CRITIC MODEL FOR TASK-COMPLETION DIALOGUE POLICY LEARNING（最后修改value学习部分可以参考这个）

[4]M. Buro, “The evolution of strong othello programs,” in Entertainment Computing - Technology and Applications, R. Nakatsu and J. Hoshino,Eds. Kluwer, 2003, pp. 81–88. 
https://skatgame.net/mburo/
https://skatgame.net/mburo/log.html