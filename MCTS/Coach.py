from collections import deque
from MCTS import MCTS # search
import utils
import numpy as np
from pytorch_classification.utils import Bar, AverageMeter #???
import time

class Coach():

    def __init__(self, env, nnet, args):
        self.env = env
        self.state = env.reset() #initial state
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.env, self.nnet, self.args) # set search tree

    def executeEpisode(self):

        trainExamples = []
        self.state = self.env.reset()
        self.curPlayer = 0
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = utils.getCanonicalForm(self.state, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = utils.getSymmetries(canonicalBoard, pi)
            for b,p in sym:
                trainExamples.append([b, self.curPlayer, p, None])  # 我们的samples里面默认是黑棋先走的状态

            action = np.random.choice(len(pi), p=pi)
            self.state, reward, done, _ = self.env.step(action)

            if done:
                self.env.render()
                return [(x[0], x[1], x[2], reward*((-1)**(x[1]!=self.curPlayer))) for x in trainExamples]

    def learn(self):

        trainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        for i in range(self.args.numIters):
            # bookkeeping
#            print('------ITER ' + str(i+1) + '------')
#            eps_time = AverageMeter()
#            bar = Bar('Self Play', max=self.args.numEps)
#            end = time.time()

            for eps in range(self.args.numEps):
                self.mcts = MCTS(self.env, self.nnet, self.args)   # reset search tree
                trainExamples += self.executeEpisode()      # append          

                # bookkeeping + plot progress
#                eps_time.update(time.time() - end)
#                end = time.time()
#                bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
#                                                                                                           total=bar.elapsed_td, eta=bar.eta_td)
#                bar.next()
#            bar.finish()

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pnet = self.nnet.__class__(self.env)
            pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.env, pnet, self.args)
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.env, self.nnet, self.args)
            ########## 要弄成AGNET

            print('PITTING AGAINST PREVIOUS VERSION')
            pwins, nwins = utils.playAGame(pmcts_agent, nmcts_agent, self.env.reset())
#                    lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
#                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
#           pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            print('NEW/PREV WINS : ' + str(nwins) + '/' + str(pwins))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet = pnet
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='checkpoint_' + str(i) + '.pth.tar')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                
