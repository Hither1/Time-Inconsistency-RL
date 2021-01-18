  
from functools import partial

import tensorflow as tf
import numpy as np
import gym
import sys


import collections
import itertools

from stable_baselines import logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.buffers import ReplayBuffer, PrioritizedReplayBuffer

from lib import plotting

'''
In this code, we replace the table look-up of hindsight distribution by a single-layer neural network
this is to retain the structure of hindsight credit assignment
'''

class StateHCA_NN(object):
  def __init__(self, n_s, n_a, hindsight_learning_rate):
    self.h_dim = n_s # number of states
    self.n_a = n_a # number of actions
    self.learning_rate = hindsight_learning_rate


    # Initialize the NN for hindsight policy
    h_one_hot = tf.one_hot(np.zeros(n_s*2, dtype=int), 3) #tf.one_hot(indices, depth)
    self.h_layer = tf.contrib.layers.fully_connected(
                  inputs=tf.expand_dims(h_one_hot, 0),
                  num_outputs=n_a,
                  activation_fn=None,
                  weights_initializer=tf.zeros_initializer)

    # Use cross-entropy as the loss function
    self.h_probs = tf.squeeze(tf.nn.softmax(self.h_layer))

    #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    #self.train_op = self.optimizer.minimize(
                #loss=cross_entropy, global_step=tf.contrib.framework.get_global_step())

    self.optimizer = tf.keras.optimizers.Adam()

    

  def train_step(self):
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    with tf.GradientTape() as tape:
      tape.watch(self.h_probs)
      loss_value = loss_object(self.h_probs, self.h_probs)

    grads = tape.gradient(loss_value, [self.h_probs, self.h_probs])

    print("Grad value is "+ str(grads))
    self.optimizer.apply_gradients(zip(grads, [tf.Variable(self.h_probs), tf.Variable(self.h_probs)]))


  def update(self, pi, V, states, rewards, gamma):
    T = len(states)
    dlogits = np.zeros_like(pi)
    dV = np.zeros_like(V)
    
  
    for i in range(T):
      x_s = states[i]
      G = ((gamma**np.arange(T - i)) * rewards[i:]).sum()
      G_hca = np.zeros(self.n_a)
      state_xs = get_state(x_s)
      
      for j in range(i, T):
        x_t, r = states[j], rewards[j]
        # hca_factor = h[:, x_s, x_t].T - pi[x_s, :] 
        # For hindsight distribution, multiply the state vector by the weights in the NN
        hca_factor = tf.linalg.matmul(tf.transpose(tf.squeeze(tf.nn.softmax(self.h_layer))), np.concatenate((x_s, x_t), axis=None).astype(np.float32).reshape(12,1)) - pi[state_xs] 
        G_hca += gamma**(j - i) * r * hca_factor # (2)

        # train h_beta via cross-entropy
        # dlogits_h[a_s, x_s, x_t] += 1
        # dlogits_h[:, x_s, x_t] -= h[:, x_s, x_t]
        self.train_step()

      for a in range(self.n_a):
        # dlogits[x_s, a] += G_hca[a]
        # dlogits[x_s] -= pi[x_s] * G_hca[a]
        dlogits[state_xs, a] += G_hca[a]
        dlogits[state_xs] -= np.multiply(pi[state_xs], G_hca[a])
        
      # dV[x_s] += (G - V[x_s])
      dV[state_xs] += (G - V[state_xs])

    return dV, dlogits

  

def hca(env, num_episodes=2000):
  
  n_s = len(env.observation_space.nvec)
  n_a = env.action_space.n
  # The learning rate is for the neural network that represents the hindsight distribution
  stateHCA = StateHCA_NN(n_s, n_a, hindsight_learning_rate=0.9)

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
  # These 13 dimenstions are for obs
  V = np.zeros(3**6, dtype=np.float16)  

  for i_episode in range(num_episodes):
    # Reset the environment and pick the first action
    state = env.reset()

    # Records of this episode
    # episode = []
    states, actions, rewards = [], [], []

    action_probs = pi[get_state(state)]
    for t in itertools.count():
      # Take a step

      action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

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
      # Get action probabilities for the new state
      action_probs = pi[get_state(state)]

    # Go through the episode and make policy updates
    # Update our policy estimator
    dlogits, dV = stateHCA.update(pi, V, states, rewards, 1.0)
    # Follow the gradients
    pi = [a + b for a, b in zip(pi, dlogits)]
    V = [a + b for a, b in zip(V, dV)]
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



#def cross_entropy(h_probs):
#  return -sum([x * np.log(x)for x in h_probs])

def get_state(state):
  res = 0
  for i in range(len(state)):
    res += state[i] * (3**i)
  return res