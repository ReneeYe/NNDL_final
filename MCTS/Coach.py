from collections import deque
from MCTS import MCTS # search
from play_game import playAGame, playGames
import agent

class Coach():   
    def __init__(self, env, nnet, args):
        self.env = env
        self.board = env.reset() #initial state
        self.nnet = nnet
        self.args = args
        self.mcts = MCTS(self.env, self.nnet, self.args) # set search tree

    def learn(self):

        trainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        for i in range(self.args.numIters):        
            self.mcts = MCTS(self.game, self.nnet, self.args)   # reset search tree
            self.mcts_agent = agent.MCTSAgent(self.mcts)
            trainExamples.append(playAGame(self.env, self.mcts_agent, self.mcts_agent, self.args.numEps, train=True))               

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pnet = self.nnet.__class__(self.env)
            pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.env, pnet, self.args)
            self.nnet.train(trainExamples)
            nmcts = MCTS(self.env, self.nnet, self.args)

            print('PITTING AGAINST PREVIOUS VERSION')   #### write a test part            
            pwins, nwins, scores = playGames(self.env, pmcts, nmcts, self.args.max_eopch, train=False)

            print('NEW/PREV WINS : ' + str(nwins) + '/' + str(pwins))
            if pwins+nwins > 0 and float(nwins)/(pwins+nwins) < self.args.updateThreshold:
                print('REJECTING NEW MODEL')
                self.nnet = pnet

            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='checkpoint_' + str(i) + '.pth.tar')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')                
