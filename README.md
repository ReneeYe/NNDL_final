# NNDL_final

reversi project

## TODO

**Fixed oppo:**

a. DQN vs. rand (已做)【求全部结果 】
b. DQN vs. greedy (TODO)

c. DQN vs. minimax （已做）
d. DQN vs. 专家 （try to open .wtb）

**self-play:**

a. 两个都是一层全连接的Q 网络，参数初始化不同（已做）【求所有结果】
b. 全连接 vs. CNN(已做，CNN还在测试)【doing by YR】

**DQN improvement**

采用基于策略函数的学习（policy gradient，actor-critic去训练，TODO）【求具体解释】
MCTS（TODO）


## 提交文件

Pdf：report
RL_QG_agent.py 猪脚的test文件，我们测的时候肯定要改要用那个存参数的
Reversi 网络参数


RL_QG_agent还需要在封装一下


## 20180114 log

可以干的活：

跑的时候记得画图/记录

~~1 把GitHub的代码改成TensorFlow的版本~~

2 可以把reward 从0/1 改成 黑-白

3 opponent的尝试： self-play，给的α-βminimax搜索（level1），找专家库（研究数据库怎么用http://www.ffothello.org/informatique/la-base-wthor/）

4 可以参考第一篇文章的参考文献中不同reward的设置

5 尝试把linear改成CNN的网络 参见第一篇论文的结构

6 想想怎么把Value函数用NN更新


## 20180115 log
TF的框架
Pytouch 没有windows版 查了差别

## 20180116 log

YR写完了NN self-play的代码

XYQ完成了部分报告

## 20180117 log

YR写CNN

XYQ写了greedy
