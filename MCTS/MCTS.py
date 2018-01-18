import math
import numpy as np

class MCTS():
    def __init__(self, env, nnet, args):
        self.env = env
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # stores game.getGameEnded ended for board s
        self.Vs = {}        # stores game.getValidMoves for board s

    def getActionProb(self, board, temp=1):

        for i in range(self.args.numMCTSSims):
            self.search(board)

        counts = [self.Nsa[(board,a)] if (board,a) in self.Nsa else 0 for a in range(self.env.board_size**2 + 1)]

        if temp==0:
            bestA = np.argmax(counts)
            probs = [0]*len(counts)
            probs[bestA]=1
            return probs

        counts = [x**(1./temp) for x in counts]
        probs = [x/float(sum(counts)) for x in counts]
        return probs

    def search(self, board):

        if board not in self.Es:
            self.Es[board] = self.game.getGameEnded(board, 1)
        if self.Es[board]!=0:
            # terminal node
            return -self.Es[board]

        if board not in self.Ps:
            # leaf node
            self.Ps[board], v = self.nnet.predict(board)
            valids = self.game.getValidMoves(board, 1)
            self.Ps[board] = self.Ps[board]*valids      # masking invalid moves
            self.Ps[board] /= np.sum(self.Ps[board])    # renormalize

            self.Vs[board] = valids
            self.Ns[board] = 0
            return -v

        valids = self.Vs[board]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (board,a) in self.Qsa:
                    u = self.Qsa[(board,a)] + self.args.cpuct*self.Ps[board][a]*math.sqrt(self.Ns[board])/(1+self.Nsa[(board,a)])
                else:
                    u = self.args.cpuct*self.Ps[board][a]*math.sqrt(self.Ns[board])     # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.env.step(board, 1, a)

        v = self.search(next_s)

        if (board,a) in self.Qsa:
            self.Qsa[(board,a)] = (self.Nsa[(board,a)]*self.Qsa[(board,a)] + v)/(self.Nsa[(board,a)]+1)
            self.Nsa[(board,a)] += 1

        else:
            self.Qsa[(board,a)] = v
            self.Nsa[(board,a)] = 1

        self.Ns[board] += 1
        return -v
