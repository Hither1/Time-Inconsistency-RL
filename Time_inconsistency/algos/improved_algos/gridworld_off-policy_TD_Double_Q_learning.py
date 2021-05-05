import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("C:\\Users\\NLESM\\Dropbox\\GradSchool\\RESEARCH\\RLImplementation\Gridworld")

from collections import defaultdict
from envs.DoughVeg_gridworld import GridworldEnv

discount_factor = 1
discounting = 'hyper' #'hyper', 'exp'
init_policy = 'random' #'random' 'stable'

