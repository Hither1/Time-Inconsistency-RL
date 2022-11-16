import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    """
    DoughVeg is created based on (cite:) to emulate the environment inducing agent's
    biasedness under hyperbolic discounting.
    o  o  V  o
    o  o  X  o
    D  o  X  o
    o  o  X  o
    o  o  X  o
    o  S  o  o
    S is your STARTing position. 
    D and V are terminal states with rewards +10 and +19, respectively. 
    X is a WALL.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge/bumping the WALL leave you in your current state.
    You receive a reward of 0 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape = [6,4]):
        
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape)
        nA = 4

        MAX_Y = shape[0]
        MAX_X = shape[1]
        
        is_done = lambda s: s == 2 or s == 8 #recurrent states
        # is_wall = lambda s: s in [6, 10, 12, 14, 16, 18, 20]
        is_wall = lambda s: s in [6, 10, 14, 18]
        
        #reward = 0.0 if is_done(s) else -1.0
        def reward(s):
            if s == 2:
                return 19
            elif s == 8:
                return 10
            else:
                return 0

        self.copy_reward_fn = reward

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags = ['multi_index'])

        while not it.finished:
            
            s = it.iterindex
            
            if is_wall(s):
                it.iternext()
                continue
            
            y, x = it.multi_index

            P[s] = {a : [] for a in range(nA)}
            
            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward(s), True)] #p, s, r, d
                P[s][RIGHT] = [(1.0, s, reward(s), True)]
                P[s][DOWN] = [(1.0, s, reward(s), True)]
                P[s][LEFT] = [(1.0, s, reward(s), True)]
            
            # Not a terminal state
            else:
                
                # Moving to a wall is not feasible; stays in its position
                ns_up = s if (y == 0 or is_wall(s - MAX_X)) else s - MAX_X
                ns_right = s if (x == (MAX_X - 1) or is_wall(s + 1)) else s + 1
                ns_down = s if (y == (MAX_Y - 1) or is_wall(s + MAX_X)) else s + MAX_X
                ns_left = s if (x == 0 or is_wall(s - 1)) else s - 1
                    
                print('s:', s)
                print('ns_up:', ns_up)
                print('ns_right:', ns_right)
                print('ns_down:', ns_down)
                print('ns_left:', ns_left)
                
                P[s][UP] = [(1.0, ns_up, reward(ns_up), is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward(ns_right), is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward(ns_down), is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward(ns_left), is_done(ns_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.ones(nS) / nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)
    
    
    '''def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "
            if x == 0:
                output = output.lstrip() 
            if x == self.shape[1] - 1:
                output = output.rstrip()
            outfile.write(output)
            if x == self.shape[1] - 1:
                outfile.write("\n")
            it.iternext()'''
