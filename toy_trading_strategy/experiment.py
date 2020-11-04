import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from HCA.hca_classes import hca # The table look-up version
#from HCA.hca_NN import hca # The NN version

import tensorflow as tf
from tradingenv import StockTradingEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

import collections
import itertools
from lib import plotting

tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)

# Set env
df = pd.read_csv('./data/AAPL.csv')
df.sort_values('Date')
df_recent = df[df['Date'] >= '2008-01-01'].reset_index()
print(df_recent)
# env = DummyVecEnv([lambda:StockTradingEnv(df)])
env = StockTradingEnv(df_recent)
observation = env.reset()

# With 

#with tf.Session() as sess:
    #sess.run(tf.initialize_all_variables())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~2000-5000 seemed to work well for me.
stats = hca(env)
print(max(stats.episode_rewards))

''' 
# Based on the results from REINFORCE, 
# hardcode a vector of the possible return states of 
ret = 0
return_bins = []
while(ret < 8000):
    return_bins.append(ret)
    ret += 500
# return-conditional
agent = ReturnHCA(n_s, n_a, return_bins)
'''