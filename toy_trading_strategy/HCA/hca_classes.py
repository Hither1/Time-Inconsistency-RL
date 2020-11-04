  
from functools import partial

import tensorflow as tf
import numpy as np
import gym
import collections
import itertools
from scipy.special import softmax

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer

from lib import plotting
# Classes for the algorithms

class StateHCA(object):
  def __init__(self, n_s, n_a):
    self.h_dim = n_s # number of states
    self.n_a = n_a # number of actions

  def update(self, pi, V, h, states, actions, rewards, gamma):
    T = len(states)
    dlogits = np.zeros_like(pi)
    dV = np.zeros_like(V)
    dlogits_h = np.zeros_like(h)
    
    for i in range(T):
      x_s, a_s = states[i], actions[i]
      G = ((gamma**np.arange(T - i)) * rewards[i:]).sum()
      G_hca = np.zeros(self.n_a)
      state_xs = get_state(x_s)
      for j in range(i, T):
        x_t, r = states[j], rewards[j]
        # hca_factor = h[:, x_s, x_t].T - pi[x_s, :] 
        state_num = get_state(np.concatenate((x_s, x_t), axis=None))
        hca_factor = [min(x, 10) for x in np.nan_to_num(np.true_divide(h[state_num], pi[state_xs]), 1)]
        
        G_hca += np.multiply(gamma**(j - i) * r, hca_factor) # (2)

        # train h_beta via cross-entropy
        # dlogits_h[a_s, x_s, x_t] += 1
        # dlogits_h[:, x_s, x_t] -= h[:, x_s, x_t]
        dlogits_h[state_num, a_s] += 1
        dlogits_h[state_num] = np.subtract(dlogits_h[state_num], h[state_num])
        
      dlogits_h /= self.n_a

      # Train probabilities
      for a in range(self.n_a):
        # dlogits[x_s, a] += G_hca[a]
        # dlogits[x_s] -= pi[x_s] * G_hca[a]
        dlogits[state_xs, a] += G_hca[a]
        dlogits[state_xs] -= np.multiply(pi[state_xs], G_hca[a])
        
      dlogits /= self.n_a
      # dV[x_s] += (G - V[x_s])
      dV[state_xs] += (G - V[state_xs])

    return dlogits, dV, dlogits_h


  

def hca(env, num_episodes=2000):
  
  n_s = len(env.observation_space.nvec)
  n_a = env.action_space.n
  # Use either algo for the model, the experiment code is the same
  HCAmodel = StateHCA(n_s, n_a)
  # HCAmodel = ReturnHCA(n_s, n_a)

  # Keeps track of useful statistics
  stats = plotting.EpisodeStats(
      episode_lengths=np.zeros(num_episodes),
      episode_rewards=np.zeros(num_episodes)) 

  # Initialize
  state = env.reset()
  
  # Initialize the policy as Uniformly Random
  # First 13 dimenstions are for obs
  # last 1 dimension is for probabilities
  pi = np.ones((3**7, 3), dtype=np.float64) / n_a
  
  # Initialize the Value function as all 0's
  # These  dimenstions are for obs
  V = np.zeros(3**6, dtype=np.float16)  

  # first n_s: current state, second n_s: any future state         
  h = np.ones((3**13, 3), dtype=np.float16) / n_a


  for i_episode in range(num_episodes):
    # Reset the environment and pick the first action
    state = env.reset()

    # Records of this episode
    # episode = []
    states, actions, rewards = [], [], []

    action_probs = pi[get_state(state)]
    print(action_probs)
    for t in itertools.count():
      # Take a step     
      action = np.random.choice(np.arange(len(action_probs)), p=[action_probs[0], action_probs[1], max(1-action_probs[0]-action_probs[1],0)])

      next_state, reward, done, _ = env.step(action)

      # Keep track of the transition
      states.append(state) 
      actions.append(action)
      rewards.append(reward) 

      # Update statistics
      stats.episode_rewards[i_episode] = max(stats.episode_rewards[i_episode], env.max_net_worth)
      stats.episode_lengths[i_episode] = t

      # Print out which step we're on, useful for debugging.
      print("\rStep {} @ Episode {}/{} ({})".format(
            t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

      if done:
              break

      state = next_state
      # Get action probabilities for the new state
      action_probs = pi[get_state(state)]

    # Go through the episode and make policy updates
    # Update our policy estimator
    dlogits, dV, dlogits_h = HCAmodel.update(pi, V, h, states, actions, rewards, gamma=0.1)
    # Follow the gradients
    pi = softmax([[a + b for a, b in zip(pi[i], dlogits[i])] for i in range(len(pi))], axis=1)
    V = [a + b for a, b in zip(V, dV)]
    h = softmax([[a + b for a, b in zip(h[i], dlogits_h[i])] for i in range(len(h))], axis=1)

  return stats
'''
class ReturnHCA(object):
  def __init__(self, n_s, n_a, return_bins):
    self.h_dim = len(return_bins)
    self._return_bins = return_bins
    
  def update(self, pi, V, h, states, actions, rewards, gamma):
    T = len(states)
    dlogits = np.zeros_like(pi)
    dV = np.zeros_like(V)
    dlogits_h = np.zeros_like(h)

    for i in range(T):
      x_s, a_s, r = states[i], actions[i], rewards[i]
      state_xs = get_state(x_s)
      G = ((gamma**np.arange(T - i)) * rewards[i:]).sum()
      G_bin_ind = (np.abs(self._return_bins - G)).argmin()
      hca_factor = (1. - pi[state_xs, :] / h[:, state_xs, G_bin_ind])
      G_hca = G * hca_factor

      dlogits[state_xs, a_s] += G_hca[a_s]
      dlogits[state_xs] -= pi[x_s] * G_hca[a_s]
      dV[x_s] += (G - V[x_s])
      dlogits_h[a_s, x_s, G_bin_ind] += 1
      dlogits_h[:, x_s, G_bin_ind] -= h[:, x_s, G_bin_ind]
        
    return dlogits, dV, dlogits_h
'''

def get_state(state):
  res = 0
  for i in range(len(state)):
    res += state[i] * (3**i)
  return res

