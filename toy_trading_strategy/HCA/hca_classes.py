  
from functools import partial

import tensorflow as tf
import numpy as np
import gym
import collections
import itertools

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
      
      for j in range(i, T):
        x_t, r = states[j], rewards[j]
        hca_factor = h[:, x_s, x_t].T - pi[x_s, :] 
        G_hca += gamma**(j - i) * r * hca_factor # (2)

        # train h_beta via cross-entropy
        dlogits_h[a_s, x_s, x_t] += 1
        dlogits_h[:, x_s, x_t] -= h[:, x_s, x_t]

      for a in range(self.n_a):
        dlogits[x_s, a] += G_hca[a]
        dlogits[x_s] -= pi[x_s] * G_hca[a]
      dV[x_s] += (G - V[x_s])

    return dlogits, dV, dlogits_h


  

def hca(env, num_episodes=2000):
  
  n_s = len(env.observation_space.nvec)
  n_a = env.action_space.n
  stateHCA = StateHCA(n_s, n_a)

  # Keeps track of useful statistics
  stats = plotting.EpisodeStats(
      episode_lengths=np.zeros(num_episodes),
      episode_rewards=np.zeros(num_episodes)) 

  # Initialize
  state = env.reset()

  policy_one_hot = tf.one_hot(state, 5)
  pi = (tf.contrib.layers.fully_connected( # Initialize the policy as Uniformly Random
                inputs=tf.expand_dims(policy_one_hot, 0),
                num_outputs=env.action_space.n,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)+1)/n_a
  pi = tf.squeeze(pi)
  

  value_one_hot = tf.one_hot(np.zeros(n_s, dtype=int), 15) # Assume to quantize the value space into 15 levels
  # because the total return seems to fall between 0 to 7500
  V = (tf.contrib.layers.fully_connected( # Initialize the Value function as all 0's
                inputs=tf.expand_dims(value_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)+1)/n_a

  state_matrix = np.tile(state, (len(state), 1))
  print(state_matrix)
  state_one_hot_matrix = tf.one_hot(state_matrix, 5)
  h = (tf.contrib.layers.fully_connected( # first n_s: current state, second n_s: any future state
                inputs=tf.expand_dims(state_one_hot_matrix, 0),
                num_outputs=3,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)+1)/n_a
  h = tf.squeeze(h)

  print(pi)
  print(h)
  for i_episode in range(num_episodes):
    # Reset the environment and pick the first action
    state = env.reset()

    # Records of this episode
    # episode = []
    states, actions, rewards = [], [], []

    action_probs = tf.reduce_sum(pi, 0)/tf.reduce_sum(pi)
    for t in itertools.count():
      # Take a step

      action = np.random.choice(np.arange(len(action_probs)), p=action_probs.numpy())
      next_state, reward, done, _ = env.step(action)

      # Keep track of the transition
      states.append(state) 
      actions.append(action)
      rewards.append(reward) 

      # Update statistics
      stats.episode_rewards[i_episode] += reward
      stats.episode_lengths[i_episode] = t

      # Print out which step we're on, useful for debugging.
      print("\rStep {} @ Episode {}/{} ({})".format(
            t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

      if done:
              break

      state = next_state

    # Go through the episode and make policy updates
    # Update our policy estimator
    dlogits, dV, dlogits_h = stateHCA.update(pi, V, h, states, actions, rewards, gamma=1.0)
    # Follow the gradients
    pi = [a + b for a, b in zip(pi, dlogits)]
    V = [a + b for a, b in zip(V, dV)]
    h = [a + b for a, b in zip(h, dlogits_h)]
    
  return stats

"""class ReturnHCA(object, n_s, n_a, num_episodes=2000):
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
      G = ((gamma**np.arange(T - i)) * rewards[i:]).sum()
      G_bin_ind = (np.abs(self._return_bins - G)).argmin()
      hca_factor = (1. - pi[x_s, :] / h[:, x_s, G_bin_ind])
      G_hca = G * hca_factor

      dlogits[x_s, a_s] += G_hca[a_s]
      dlogits[x_s] -= pi[x_s] * G_hca[a_s]
      dV[x_s] += (G - V[x_s])
      dlogits_h[a_s, x_s, G_bin_ind] += 1
      dlogits_h[:, x_s, G_bin_ind] -= h[:, x_s, G_bin_ind]
        
    return dlogits, dV, dlogits_h

"""