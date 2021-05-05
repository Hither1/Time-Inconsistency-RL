"""
Created on Mon Feb 2

@author: Huangyuan

Policy ---> agent in original paper

"""

from envs.DoughVeg_gridworld import GridworldEnv
#from envs.DoughVeg_windy import GridworldEnv
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.special import softmax
import time

current_env_windy = False  # Change between normal/windy gridworlds

discount_factor = 1
discounting = 'hyper'  # 'hyper', 'exp'
init_policy = 'random'  # 'random' 'stable'

alpha = 0.3  # The noise parameter that modulates between random choice (=0) and perfect maximization (=\infty)
#epsilon = .6
num_episodes = 20000  # 0000

env = GridworldEnv()

if current_env_windy:
    is_wall = lambda s: s in [6, 13, 20, 27, 35, 36, 37, 38, 39]  # windy gridworld
else:
    is_wall = lambda s: s in [6, 10, 14, 18]


def auto_discounting(discount_factor=discount_factor):
    def hyper(i, discount_factor=discount_factor):
        return 1 / (1 + discount_factor * i)

    def exp(i, discount_factor=discount_factor):
        return discount_factor ** i

    if discounting == 'hyper':
        return hyper
    else:
        return exp


def make_policy(expUtility, alpha):
    """
    Creates an probabilistic policy based on a given expected Utility function and alpha.

    Args:
        expUtility: A dictionary that maps from (state, delay) -> action-values.
            Each value is a numpy array of length nA (see below)
        alpha: moderating parameter

    Returns:
        A function that takes the observation(state, delay) as an argument and returns
        an action according to the probabilities for each action.

    """

    def policy_fn(state, delay):
        return softmax(expUtility[state][delay] * alpha)

    return policy_fn


# Hyperbolic Discounted Q-learning (off-policy TD control)
def td_control(env, num_episodes, step_size):

    global expu_correction_21, expu_u, expu_r, expu_b, expu_l

    # The type of discounting
    discount = auto_discounting()

    # Global variables
    global critical_episode_9, critical_episode_21

    # Initialize the expUtility matrix
    # We store the values in the order expUtility[state][delay][utility]
    expUtility = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
    for a_ in range(4):
        for d in range(15):
            expUtility[8][d][a_] = 10
            expUtility[2][d][a_] = 19

    # Take the Utility function from reward function of the environment
    Utility = env.copy_reward_fn

    # Agent (Policy) when given expUtility(state, delay, *)
    agent = make_policy(expUtility, alpha)  # , env.action_space.n)

    for i_episode in range(1, num_episodes + 1):
        print("state 21", expUtility[21][0])
        print("state 1", expUtility[1])
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()

        if is_wall(state):
            # print('init on wall, skip')
            continue

        for t in range(100):
            # print("current state " + str(state))

            probs = agent(state, t)  # Select an action according to policy
            if current_env_windy:
                if state == 31:
                    action = 0
                else:
                    action = np.random.choice(np.arange(len(probs)), p=probs)
            else:
                action = np.random.choice(np.arange(len(probs)), p=probs)

            next_state, reward, done, _ = env.step(action)

            episode.append((state, action, reward))

            # Update q_value for a state-action pair Q(s,a):
            # Q(s,a) = Q(s,a) + α( r + γmaxa' Q(s',a') - Q(s,a) )

            # Do the computation
            u = discount(t+1) * Utility(state)
            expectation = np.dot(expUtility[next_state][t + 1],
                                 softmax(expUtility[next_state][t + 1] * alpha).T)

            expUtility[state][t][action] = u + expectation

            if state == 9 and action == 3:
                print('----')
                print('state:', state)
                print('probs:', probs)
                print('action:', action)
                print('next_state:', next_state)
                print('reward:', reward)
                print('done:', done)
                print('u:', u)
                print('expectation:', expectation)

                print('t+1', t + 1)
                for a_ in range(4):
                    print('a_:', a_)
                    print('expUtility[next_state]',
                          [expUtility[next_state][d_][a_] for d_ in expUtility[next_state].keys()])

            if t == 0 and state in [2, 8]:
                print('----')
                print('state:', state)
                print('probs:', probs)
                print('action:', action)
                print('next_state:', next_state)
                print('reward:', reward)
                print('done:', done)
                # print('u:', u)
                print('expectation:', expectation)

            if u != 0:
                print("expU", expUtility[state][t])

            # if final state
            if done:
                break
            state = next_state  # update to the next state

            if state == 9 and action == 0:  # check for possible JUMP at 9 due to noisy policy

                if critical_episode_9 == 0:
                    critical_episode_9 = i_episode - 1

                # count_9 += 1

                curr_policy = np.argmax(agent(state, 0))
                curr_Q = expUtility[state][0]  # delay = 0
                print('curr_policy_9:', curr_policy)
                print('curr_Q[9]:', curr_Q)

                print('-----')

            # if state == 21 and action == 1: # check for when the change to SPE is reflected at 21
            if state == 21 and expUtility[21][0][1] > expUtility[21][0][0]:

                if critical_episode_21 == 0:  # update the episode in which we reach state 21
                    critical_episode_21 = i_episode - 1

                # count_21 += 1

                # curr_policy = np.argmax(agent(state))
                curr_Q = expUtility[state][0]
                # print('curr_policy:', curr_policy)
                print('curr_Q[21]:', curr_Q)

                # print('trajectory aft 21:', episode[first_occurence_idx:])

                print('--------------')
        if current_env_windy:
            # Track (Q[24], Q[25], Q[31], Q[32]) for all actions and plot
            expu_u.append([expUtility[24][0][0], expUtility[25][0][0], expUtility[31][0][0], expUtility[32][0][0]])
            expu_r.append([expUtility[24][0][1], expUtility[25][0][1], expUtility[31][0][1], expUtility[32][0][1]])
            expu_b.append([expUtility[24][0][2], expUtility[25][0][2], expUtility[31][0][2], expUtility[32][0][2]])
            expu_l.append([expUtility[24][0][3], expUtility[25][0][3], expUtility[31][0][3], expUtility[32][0][3]])
        else:
            # Track (Q[21], Q[9]) for all actions and plot
            expu_u.append([expUtility[21][0][0], expUtility[9][0][0]])
            expu_r.append([expUtility[21][0][1], expUtility[9][0][1]])
            expu_b.append([expUtility[21][0][2], expUtility[9][0][2]])
            expu_l.append([expUtility[21][0][3], expUtility[9][0][3]])


    return expUtility


# Critical to check how sensitive is relevant states (i.e. 13, 17, 21) to sudden deviation at 9
critical_states_update_order = []
critical_index_9 = 0
critical_episode_9 = 0
critical_index_21 = 0
critical_episode_21 = 0

# Initialize the arrays to store values of Q


# Critical to check how sensitive is relevant states (i.e. 13, 17, 21) to sudden deviation at 9
critical_states_update_order = []
critical_index_9 = 0
critical_episode_9 = 0
critical_index_21 = 0
critical_episode_21 = 0

Q_correction_21 = []
expu_u = []
expu_r = []
expu_b = []
expu_l = []

np.random.seed(0)

start = time.time()
expUtility = td_control(env, num_episodes, step_size=0.01)
end = time.time()

print(expUtility[21].keys())
print([expUtility[21][d][0] for d in expUtility[21].keys()])
print([expUtility[21][d][1] for d in expUtility[21].keys()])

print(expUtility[9].keys())
print([expUtility[9][d][0] for d in expUtility[21].keys()])
print([expUtility[9][d][1] for d in expUtility[21].keys()])

# ------------------------------------------------------------------------------------------------

'''Checking criticals'''
#print('EPSILON:', epsilon)
print('POLICY INIT:', init_policy)
print('DISCOUNTING:', discounting)
print('-----')
print('(9, UP) first appears at ep:', critical_episode_9)
print('visitation to 21 aft:', critical_states_update_order[critical_index_9:].count(21))
# print('(21, RIGHT) first appears at ep:', critical_episode_21)
print('Q(21, RIGHT) > Q(21, UP) first appears at ep:', critical_episode_21)
print('Time used: ', end - start)


if current_env_windy:
    x = [i for i in range(1, 1 + len(expu_u))]
    # first pic
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, np.array(np.array(expu_r)[:, 0]) - np.array(np.array(expu_l)[:, 0]))
    axs[0, 0].set_title('Q(24, r) - Q(24, l) (Naive)')
    axs[0, 1].plot(x, np.array(np.array(expu_b)[:, 1]) - np.array(np.array(expu_u)[:, 1]))
    axs[0, 1].set_title('Q(25, b) - Q(25, u) (Naive)')
    axs[1, 0].plot(x, np.array(np.array(expu_l)[:, 2]) - np.array(np.array(expu_u)[:, 2]))
    axs[1, 0].set_title('Q(31, l) - Q(31, u) (Naive)')
    axs[1, 1].plot(x, np.array(np.array(expu_u)[:, 3]) - np.array(np.array(expu_l)[:, 3]))
    axs[1, 1].set_title('Q(32, u) - Q(32, l) (Naive)')

    fig.show()

    # second pic
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, np.array(expu_u)[:, 0], label='u')
    axs[0, 0].plot(x, np.array(expu_r)[:, 0], label='r')
    axs[0, 0].plot(x, np.array(expu_b)[:, 0], label='b')
    axs[0, 0].plot(x, np.array(expu_l)[:, 0], label='l')
    axs[0, 0].set_title('State 24 (Naive)')
    axs[0, 0].legend()
    axs[0, 1].plot(x, np.array(expu_u)[:, 1], label='u')
    axs[0, 1].plot(x, np.array(expu_r)[:, 1], label='r')
    axs[0, 1].plot(x, np.array(expu_b)[:, 1], label='b')
    axs[0, 1].plot(x, np.array(expu_l)[:, 1], label='l')
    axs[0, 1].set_title('State 25 (Naive)')
    axs[0, 1].legend()
    axs[1, 0].plot(x, np.array(expu_u)[:, 2], label='u')
    axs[1, 0].plot(x, np.array(expu_r)[:, 2], label='r')
    axs[1, 0].plot(x, np.array(expu_b)[:, 2], label='b')
    axs[1, 0].plot(x, np.array(expu_l)[:, 2], label='l')
    axs[1, 0].set_title('State 31 (Naive)')
    axs[1, 0].legend()
    axs[1, 1].plot(x, np.array(expu_u)[:, 3], label='u')
    axs[1, 1].plot(x, np.array(expu_r)[:, 3], label='r')
    axs[1, 1].plot(x, np.array(expu_b)[:, 3], label='b')
    axs[1, 1].plot(x, np.array(expu_l)[:, 3], label='l')
    axs[1, 1].set_title('State 32 (Naive)')
    axs[1, 1].legend()

    plt.show()
else:
    x = [i for i in range(1, 1 + len(expu_u))]
    # first pic
    fig, axs = plt.subplots(1, 2)
    print("Final Q_u")
    print(expu_u)
    axs[0].plot(x, np.array(expu_u)[:, 0] - np.array(expu_r)[:, 0])
    axs[0].set_title('Difference Q(21, u) - Q(21, r) (Naive)')
    axs[1].plot(x, np.array(expu_l)[:, 1] - np.array(expu_u)[:, 1])
    axs[1].set_title('Difference Q(9, l) - Q(9, u) (Naive)')
    fig.show()

    # second pic
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x, np.array(expu_u)[:, 0], label='u')
    axs[0].plot(x, np.array(expu_r)[:, 0], label='r')
    axs[0].plot(x, np.array(expu_b)[:, 0], label='b')
    axs[0].plot(x, np.array(expu_l)[:, 0], label='l')
    axs[0].set_title('Q(s=21) Naive: Gridworld')
    axs[0].legend()
    axs[1].plot(x, np.array(expu_u)[:, 1], label='u')
    axs[1].plot(x, np.array(expu_r)[:, 1], label='r')
    axs[1].plot(x, np.array(expu_b)[:, 1], label='b')
    axs[1].plot(x, np.array(expu_l)[:, 1], label='l')
    axs[1].set_title('Q(s=9) Naive: Gridworld')
    plt.legend()
    fig.show()


