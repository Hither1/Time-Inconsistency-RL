# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:37:29 2022

@author: Nixie S Lesmana
"""

import numpy as np
import matplotlib.pyplot as plt

#sys.path.append("")

from collections import defaultdict

import sys
import os

cwd = os.getcwd()
sys.path.append(cwd) # nb: append cwd or "../" not enough, need to add __init__ to the cwd;
if "../" not in sys.path:
    sys.path.append("../")

env_ID = 'S'

if env_ID == 'D':   
    from lib.envs.DoughVeg_gridworld import GridworldEnv # stochastic simple gridworld
else:
    from lib.envs.DoughVeg_simple_stochastic import GridworldEnv

#from envs.DoughVeg_gridworld import GridworldEnv

discount_factor = 1
discounting = 'hyper' #'hyper', 'exp'
init_policy = 'random' #'random' 'stable'

epsilon = .07
num_episodes = 30000 #0000
num_seeds = 50

env = GridworldEnv()

is_wall = lambda s: s in [6, 10, 14, 18]

def auto_discounting(discount_factor = discount_factor):
        
    def hyper(i, discount_factor = discount_factor):
        return 1/(1 + discount_factor*i)

    def exp(i, discount_factor = discount_factor):
        return discount_factor**i
    
    if discounting == 'hyper':
        return hyper
    else:
        return exp

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        
        return A
    
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, discount_factor, epsilon):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    global critical_states_update_order, critical_index_9, critical_episode_9, critical_index_21, critical_episode_21
    global q_u, q_r, q_b, q_l
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # policy is one-to-one to Q: init at UP for each s
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    for a in range(env.action_space.n):
        Q[2][a] = 19
        Q[8][a] = 10
    # Stable policy initialization
    if init_policy == 'stable':
        policy(9)[0] = epsilon/4
        policy(9)[3] += 1 - epsilon        
    
    # The type of discounting
    discount = auto_discounting()
    
    for i_episode in range(1, num_episodes + 1):
        
        #Debugging
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = []
        state = env.reset() #s0 is randomized uniformly

        while is_wall(state) or state == 2 or state == 8:
            state = env.reset()
        
        # Asynchronous kind
        s0 = state

        for t in range(100):
            
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
                        
            episode.append((state, action, reward))
            
            if done:
                break
            
            state = next_state

            
        # Find all (state, action) pairs we've visited in this episode
        # We convert each state to a tuple so that we can use it as a dict key
        sa_in_episode = set([(x[0], x[1]) for x in episode])
        
        for state, action in sa_in_episode:
            
            sa_pair = (state, action)
            
            # Find the first occurence of the (s, a) pair (FROM THE FRONT, correct.)
            # This update is questionable for INCONSISTENT problem.
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)
            
            G = sum([x[2]*(discount(i+1)) for i, x in enumerate(episode[first_occurence_idx:])])
            
            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0
            

                
            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]
            

            
            if state == 9 and action == 0: # check for possible JUMP at 9 due to noisy policy
                
                if critical_episode_9 == 0:
                    critical_episode_9 = i_episode - 1
                
                #count_9 += 1
                
                curr_policy = np.argmax(policy(state))
                curr_Q = Q[state]
                #print('curr_policy_9:', curr_policy)
                #print('curr_Q[9]:', curr_Q)
                
                #print('G:', G)
                

                
            #if state == 21 and action == 1: # check for when the change to SPE is reflected at 21
            if state == 21 and Q[21][1] > Q[21][0]:
                
                if critical_episode_21 == 0: # update the episode in which we reach state 21
                    critical_episode_21 = i_episode - 1
                    
                #count_21 += 1
                
                curr_policy = np.argmax(policy(state))
                curr_Q = Q[state]
                
                #print('trajectory aft 21:', episode[first_occurence_idx:])
                
                #print('--------------')
            
        # Track Q[21] for all actions and plot
        q_u.append([Q[21][0], Q[9][0]])
        q_r.append([Q[21][1], Q[9][1]])
        q_b.append([Q[21][2], Q[9][2]])
        q_l.append([Q[21][3], Q[9][3]])
        
    return Q

# Critical to check how sensitive is relevant states (i.e. 13, 17, 21) to sudden deviation at 9
critical_states_update_order = []
critical_index_9 = 0
critical_episode_9 = 0
critical_index_21 = 0
critical_episode_21 = 0

Q_correction_21 = []

np.random.seed(0)
q_u_s = []
q_r_s = []
q_b_s = []
q_l_s = []
for _ in range(num_seeds):
    revisits = []
    q_correction_21 = []
    q_u = []
    q_r = []
    q_b = []
    q_l = []
    Q = mc_control_epsilon_greedy(env, num_episodes, discount_factor, epsilon)
    q_u_s.append(q_u)
    q_r_s.append(q_r)
    q_b_s.append(q_b)
    q_l_s.append(q_l)

q_u_s = np.array(q_u_s)
q_r_s = np.array(q_r_s)
q_b_s = np.array(q_b_s)
q_l_s = np.array(q_l_s)

final_q_u = []
final_q_r = []
final_q_b = []
final_q_l = []
for i in range(num_episodes):
    final_q_u.append([np.mean(q_u_s[:, i, 0]), np.std(q_u_s[:, i, 0]), np.mean(q_u_s[:, i, 1]), np.std(q_u_s[:, i, 1])])
    final_q_r.append([np.mean(q_r_s[:, i, 0]), np.std(q_r_s[:, i, 0]), np.mean(q_r_s[:, i, 1]), np.std(q_u_s[:, i, 1])])
    final_q_b.append([np.mean(q_b_s[:, i, 0]), np.std(q_b_s[:, i, 0]), np.mean(q_b_s[:, i, 1]), np.std(q_u_s[:, i, 1])])
    final_q_l.append([np.mean(q_l_s[:, i, 0]), np.std(q_l_s[:, i, 0]), np.mean(q_l_s[:, i, 1]), np.std(q_u_s[:, i, 1])])
final_q_u = np.array(final_q_u)
final_q_r = np.array(final_q_r)
final_q_b = np.array(final_q_b)
final_q_l = np.array(final_q_l)
#------------------------------------------------------------------------------------------------

'''Checking criticals'''
print('EPSILON:', epsilon)
print('POLICY INIT:', init_policy)
print('DISCOUNTING:', discounting)
print('-----')
print('(9, UP) first appears at ep:', critical_episode_9)
print('visitation to 21 aft:', critical_states_update_order[critical_index_9:].count(21))
#print('(21, RIGHT) first appears at ep:', critical_episode_21)
print('Q(21, RIGHT) > Q(21, UP) first appears at ep:', critical_episode_21)

print("Final Best Actions:")
nr = env.shape[0]
nc = env.shape[1]
for r_ in range(nr):
    row = []
    for c_ in range(nc):
        row.append(np.argmax(Q[r_ * nc + c_]))
    print(row)
# Graphs
x = [i for i in range(1, 1 + len(q_u))]
# first pic
fig, axs = plt.subplots(1, 2)
print("Final Q_u")
# print(q_u)
axs[0].plot(x, final_q_u[:, 0] - final_q_r[:, 0])
axs[0].set_title('Difference Q(21, u) - Q(21, r)')
axs[1].plot(x, final_q_l[:, 2] - final_q_u[:, 2])
axs[1].set_title('Difference Q(9, l) - Q(9, u)')
#MC: \epsilon-greedy (\epsilon = .2)
fig.suptitle('MC: \u03B5-greedy' + ' (\u03B5=' + str(epsilon)+')')
fig.show()

# second pic
fig, axs = plt.subplots(1, 2)
axs[0].plot(x, final_q_u[:, 0], label='u')
axs[0].fill_between(x, final_q_u[:, 0] - final_q_u[:, 1], final_q_u[:, 0] + final_q_u[:, 1], alpha=0.2)
print("(21, up)", np.array(q_u)[:, 0][-1])
axs[0].plot(x, final_q_r[:, 0], label='r')
axs[0].fill_between(x, final_q_r[:, 0] - final_q_r[:, 1], final_q_r[:, 0] + final_q_r[:, 1], alpha=0.2)
print("(21, right)", np.array(q_r)[:, 0][-1])
axs[0].plot(x, final_q_b[:, 0], label='b')
axs[0].fill_between(x, final_q_b[:, 0] - final_q_b[:, 1], final_q_b[:, 0] + final_q_b[:, 1], alpha=0.2)
print("(21, below)", np.array(q_b)[:, 0][-1])
axs[0].plot(x, final_q_l[:, 0], label='l')
axs[0].fill_between(x, final_q_l[:, 0] - final_q_l[:, 1], final_q_l[:, 0] + final_q_l[:, 1], alpha=0.2)
print("(21, left)", np.array(q_l)[:, 0][-1])
axs[0].set_title('Q(s=21) Gridworld')
axs[0].legend()

axs[1].plot(x, final_q_u[:, 2], label='u')
axs[1].fill_between(x, final_q_u[:, 2] - final_q_u[:, 3], final_q_u[:, 2] + final_q_u[:, 3], alpha=0.2)
axs[1].plot(x, final_q_r[:, 2], label='r')
axs[1].fill_between(x, final_q_r[:, 2] - final_q_r[:, 3], final_q_r[:, 2] + final_q_r[:, 3], alpha=0.2)
axs[1].plot(x, final_q_b[:, 2], label='b')
axs[1].fill_between(x, final_q_b[:, 2] - final_q_b[:, 3], final_q_b[:, 2] + final_q_b[:, 3], alpha=0.2)
axs[1].plot(x, final_q_l[:, 2], label='l')
axs[1].fill_between(x, final_q_l[:, 2] - final_q_l[:, 3], final_q_l[:, 2] + final_q_l[:, 3], alpha=0.2)
axs[1].set_title('Q(s=9) Gridworld')
plt.legend()
fig.suptitle('MC: \u03B5-greedy' + ' (\u03B5=' + str(epsilon)+')')
fig.show()


'''Computing V*(s)'''
V_estimate = defaultdict(float)

for state, actions in Q.items():
    action_value = np.max(actions)
    V_estimate[state] = action_value

# Check any scaling issue
avg_V_est = np.average([V_estimate[s] for s in V_estimate.keys()])

'''Computing real \pi*-driven-trajectory values R(\tau)
V_realized = defaultdict(float)
V_error = defaultdict(float)
discount = auto_discounting(discount_factor)
for s0 in range(24):
    
    if is_wall(s0):
        continue
    # reset env manually
    env.s = s0
    env.lastaction = None
    
    episode = []
    state = s0
    for t in range(100):
        
        probs = policy(state)
        action = np.random.choice(np.arange(len(probs)), p=probs)
        next_state, reward, done, _ = env.step(action)
                    
        episode.append((state, action, reward))
        
        if done:
            break
        
        state = next_state
        
    #Total_Reward = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode)])
    Total_Reward = sum([x[2]*(discount(i)) for i, x in enumerate(episode)])
    
    if s0 == 21:
        print(episode)
        print(Total_Reward)
        
    V_realized[s0] = Total_Reward
    V_error[s0] = Total_Reward - V_estimate[s0]
# Check any scaling issue
avg_V_real = np.average([V_realized[s] for s in V_realized.keys()])'''

''' Results
print('DISCOUNTING:', discounting)
print('sq_error of value_estimate:', V_error)
print('sum_of_sq_error:', sum([V_error[s]**2 for s in V_error.keys()]))
print('averaged values:', avg_V_real)
'''