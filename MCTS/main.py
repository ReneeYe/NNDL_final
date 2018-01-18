from Coach import Coach
from NNet import NNet as nn
from utils import *
import gym

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,
    'arenaCompare': 40,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
})



if __name__=="__main__":
    env = gym.make('Reversi8x8-v0')
    nnet = nn(env) 
    
    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(env, nnet, args)#set train para
    c.learn()#train
