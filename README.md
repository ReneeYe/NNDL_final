# NNDL_final

reversi project

## TODO

**Fixed oppo:**

a. DQN vs. rand (已做)

b. DQN vs. greedy (已做)

c. DQN vs. minimax （已做）

d. DQN vs. 专家对局库 （try to open .wtb）


**self-play:**

a. 两个都是一层全连接的Q 网络，参数初始化不同（已做）【求所有结果】
b. CNN网络（已做）

**DQN improvement**

CNN-DQN加入MCTS（程序已写完，还需调参，跑QAQ）


## Conclussion
### basline
rand, greedy, minimax(alphabeta prun)

### metric
比赛50局

### table

| ////////   |Fixed		  |Self-play					           		|  		    

|----------	|--------------------------------	|

|//////// |FC-minimax	|FC-DQN		  | CNN-DQN  |MCTS-CNN	|

|-------  |----------	|----------	|--------  |----------|

| Rnd		  |			./50  |	    ./50  |    ./50  |    ./50  |

| Greedy	|			./50  |    ./50   |    ./50  |    ./50  |

| Minimax	|			./20  |    ./50   |    ./50  |    ./50  |


## 提交文件

Pdf：report

RL_QG_agent.py 助教的test文件，我们测的时候肯定要改要用那个存参数的

Reversi 网络参数

RL_QG_agent还需要在封装一下


## 20180114 log

可以干的活：

1 跑的时候记得画图/记录

2 可以把reward 从0/1 改成 黑-白

3 opponent的尝试： self-play，给的α-βminimax搜索（level1），找专家库（研究数据库怎么用http://www.ffothello.org/informatique/la-base-wthor/）

4 可以参考第一篇文章的参考文献中不同reward的设置

5 ~~尝试把FC（dense）改成CNN的网络 参见第一篇论文的结构~~

6 ~~想想怎么把Value函数用NN更新~~

7 ~~棋盘rotation以及对称操作加入经验池去训练~~

8 MCTS-CNN

## 20180115 log

TF的框架

## 20180116 log

YR写完了fcNN self-play的代码

XYQ完成了部分报告

## 20180117 log

YR写CNN

XYQ写了greedy

## 20180118 log
写完了MCTS，要调参
