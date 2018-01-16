# report有关算法的主体部分的框架

## 1. 学习下棋的整体构架
###	(1) fixed-opponent：即对方的策略是给定的
		a. DQN vs. rand (已做)
		b. DQN vs. greedy (a greedy player choose a modve that causes the maximum number of filps in the next step, TODO)
		c. DQN vs. minimax （TODO）
		d. DQN vs. 专家 （TODO）

###	（2）self-play：即对方的策略也是需要学习的
		a. 两个都是一层全连接的Q 网络，参数初始化不同（已做）
		b. 全连接 vs. CNN(等，就是在Q网络上做文章)

## 2. 处理策略学习（即深度Q网络，DQN）的一些tricks
###	（1）用Q learning的Q函数估计（就是P250下面的公式）
###	（2）引入参数phi，即近似估计Qlearning中的Q函数
###	（3）为解决参数学习中目标也依赖参数导致样本有很强相关性，采样经验回放算法（experience replay），即构建一个经验池，去除数据相关性（书P252-253和我的demo.docx）
###	（4）采用基于策略函数的学习（policy gradient，actor-critic去训练，TODO）

