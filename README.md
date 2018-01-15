# NNDL_final

reversi project

## 20180114 log
https://github.com/Gosicfly/Reversi_PolicyGradient/blob/master/

可以干的活：

跑的时候记得画图/记录

1 把GitHub的代码改成TensorFlow的版本

2 可以把reward 从0/1 改成 黑-白

3 opponent的尝试： self-play，给的α-βminimax搜索（level1），找专家库（研究数据库怎么用http://www.ffothello.org/informatique/la-base-wthor/）

4 可以参考第一篇文章的参考文献中不同reward的设置

5 尝试把linear改成CNN的网络 参见第一篇论文的结构

6 想想怎么把Value函数用NN更新

## 参考文献
【1】Learning to Play Othello with Deep Neural Networks（CNN的那篇）

【2】Reinforcement Learning in the Game of Othello: Learning Against a Fixed Opponent and Learning from Self-Play（主要参考程序章结构）

【3】ADVERSARIAL ADVANTAGE ACTOR-CRITIC MODEL FOR TASK-COMPLETION DIALOGUE POLICY LEARNING（最后修改value学习部分可以参考这个）

## 提交文件
	Pdf：report
	RL_QG_agent.py 猪脚的test文件，我们测的时候肯定要改要用那个存参数的
	Reversi 网络参数


## 20180115
TF的框架
Pytouch 没有windows版 查了差别
