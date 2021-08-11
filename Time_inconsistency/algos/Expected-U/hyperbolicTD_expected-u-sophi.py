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
import pylab as pl

current_env_windy = False
#current_env_windy = True  # Change between normal/windy gridworlds
isSoftmax = True

discount_factor = 1
discounting = 'hyper'  # 'hyper', 'exp'
init_policy = 'random'  # 'random' 'stable'

# The noise parameter that modulates between random choice (=0) and perfect maximization (=\infty)

num_episodes = 30000

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

if isSoftmax:
    alpha = .5  # The noise parameter that modulates between random choice (=0) and perfect maximization (=\infty)
else:
    epsilon = .28

def make_policy(expUtility, nA, isSoftmax):
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
    def policy_fn(state):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(expUtility[state][0])
        A[best_action] += (1.0 - epsilon)
        return A

    def policy_fn_softmax(state):
        return softmax(expUtility[state][0] * alpha)

    if isSoftmax:
        return policy_fn_softmax
    else:
        return policy_fn


# Hyperbolic Discounted Q-learning (off-policy TD control)
def td_control(env, num_episodes, isSoftmax, step_size):
    global expu_correction_21, expu_u, expu_r, expu_b, expu_l

    # The type of discounting
    discount = auto_discounting()

    # Global variables
    global critical_episode_9, critical_episode_21
    global revisits

    # Initialize the expUtility matrix
    # We store the values in the order expUtility[state][delay][utility]
    expUtility = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
    if current_env_windy:
        for a_ in range(4):
            for d in range(100):
                expUtility[4][d][a_] = 6
                expUtility[10][d][a_] = 3
                expUtility[28][d][a_] = 10

    else:
        for a_ in range(4):
            for d in range(15):
                expUtility[8][d][a_] = 10
                expUtility[2][d][a_] = 19


    # Take the Utility function from reward function of the environment
    if current_env_windy:
        Utility = defaultdict(lambda: 0)
        Utility[4] = 6
        Utility[10] = 3
        Utility[28] = 10
    else:
        Utility = env.copy_reward_fn

    # Agent (Policy) when given expUtility(state, delay, *)
    agent = make_policy(expUtility, env.action_space.n, isSoftmax=isSoftmax)

    for i_episode in range(1, num_episodes + 1):
        states = set({})  # keep a set of states we already visited

        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # Generate an episode.
        # An episode is an array of (state, action, reward) tuples
        episode = []
        state = env.reset()

        while is_wall(state) or state == 2 or state == 8:
            state = env.reset()

        current_revisit = 0
        for t in range(100):
            if state in states:
                current_revisit += 1
            else:
                states.add(state)

            probs = agent(state)  # Select an action according to policy

            action = np.random.choice(np.arange(len(probs)), p=probs)

            next_state, reward, done, _ = env.step(action)

            if next_state != state:
                episode.append((state, action, reward))

            # Update q_value for a state-action pair Q(s,a):
            # Q(s,a) = Q(s,a) + α( r + γmaxa' Q(s',a') - Q(s,a) )

            # Do the computation
            state_utility = Utility[state] if current_env_windy else Utility(state)
            u = discount(t) * state_utility
            if isSoftmax:
                x = np.dot(expUtility[next_state][t + 1],
                                 softmax(expUtility[next_state][t + 1] * alpha).T)
                expectation = x # np.log(x) if x > 0 else x
                print("Sum of softmax", sum(softmax(expUtility[next_state][t + 1])))
                #next_action = np.random.choice(np.arange(len(probs)), p=agent(next_state))
                #expectation = expUtility[next_state][t+1][next_action]
            else:
                expectation = np.dot(expUtility[next_state][t + 1], agent(next_state).T)

            #expUtility[state][t][action] = u + expectation
            expUtility[state][t][action] = (1 - step_size) * expUtility[state][t][action] + step_size* (u + expectation)


            if done:
                break
            state = next_state  # update to the next state

            if state == 9 and action == 0:  # check for possible JUMP at 9 due to noisy policy

                if critical_episode_9 == 0:
                    critical_episode_9 = i_episode - 1

            # if state == 21 and action == 1: # check for when the change to SPE is reflected at 21
            if state == 21 and expUtility[21][0][1] > expUtility[21][0][0]:

                if critical_episode_21 == 0:  # update the episode in which we reach state 21
                    critical_episode_21 = i_episode - 1


        print("Episode", i_episode)
        print(episode)
        revisits.append(current_revisit)
        if current_env_windy:
            # Track Q[24] for all actions and plot
            expu_u.append([expUtility[24][0][0], expUtility[25][0][0], expUtility[31][0][0], expUtility[32][0][0]])
            expu_r.append([expUtility[24][0][1], expUtility[25][0][1], expUtility[31][0][1], expUtility[32][0][1]])
            expu_b.append([expUtility[24][0][2], expUtility[25][0][2], expUtility[31][0][2], expUtility[32][0][2]])
            expu_l.append([expUtility[24][0][3], expUtility[25][0][3], expUtility[31][0][3], expUtility[32][0][3]])
        else:
            # Track Q[21] for all actions and plot
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

revisits = []
np.random.seed(0)
q_u_s = []
q_r_s = []
q_b_s = []
q_l_s = []
for _ in range(5):
    expu_correction_21 = []
    expu_u = []
    expu_r = []
    expu_b = []
    expu_l = []
    expUtility = td_control(env, num_episodes, isSoftmax=isSoftmax, step_size=.5)
    q_u_s.append(expu_u)
    q_r_s.append(expu_r)
    q_b_s.append(expu_b)
    q_l_s.append(expu_l)
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
# ------------------------------------------------------------------------------------------------

'''Checking criticals'''
print('-----')
print('(9, UP) first appears at ep:', critical_episode_9)
print('visitation to 21 aft:', critical_states_update_order[critical_index_9:].count(21))
# print('(21, RIGHT) first appears at ep:', critical_episode_21)
print('Q(21, RIGHT) > Q(21, UP) first appears at ep:', critical_episode_21)

print("Average number of times the agent has revisited states",
      sum(revisits) / len(revisits))
print("Final Best Actions:")
nr = env.shape[0]
nc = env.shape[1]
for r_ in range(nr):
    row = []
    for c_ in range(nc):
        row.append(np.argmax(expUtility[r_ * nc + c_][0]))
    print(row)

if current_env_windy:
    print("Final Best Actions with Wind:")
    for r_ in range(nr):
        row = []
        for c_ in range(nc):
            if r_ * nc + c_ == 24 or r_ * nc + c_ == 31:
                a = 0 if expUtility[r_ * nc + c_][0][0] > expUtility[r_ * nc + c_][0][1] else 1
                row.append(a)
            else:
                row.append(np.argmax(expUtility[r_ * nc + c_][0]))
        print(row)

    x = [i for i in range(1, 1 + len(expu_u))]
    # first pic
    fig, axs = plt.subplots(2, 2)
    pl.subplots_adjust(wspace=.3, hspace=.3)

    axs[0, 0].plot(x, np.array(np.array(expu_u)[:, 0]) - np.array(np.array(expu_r)[:, 0]))
    axs[0, 0].set_title('Q(24, u) - Q(24, r) (Soph.)')
    axs[0, 1].plot(x, np.array(np.array(expu_u)[:, 1]) - np.array(np.array(expu_b)[:, 1]))
    axs[0, 1].set_title('Q(25, u) - Q(25, b) (Soph.)')
    axs[1, 0].plot(x, np.array(np.array(expu_l)[:, 2]) - np.array(np.array(expu_u)[:, 2]))
    axs[1, 0].set_title('Q(31, l) - Q(31, u) (Soph.)')
    axs[1, 1].plot(x, np.array(np.array(expu_u)[:, 3]) - np.array(np.array(expu_l)[:, 3]))
    axs[1, 1].set_title('Q(32, u) - Q(32, l) (Soph.)')
    if isSoftmax:
        fig.suptitle('Forward: Using Softmax' + ' alpha: ' + str(alpha))
    else:
        fig.suptitle('Forward: \u03B5-greedy' + ' (\u03B5=' + str(epsilon)+')')
    fig.show()

    # second pic
    fig, axs = plt.subplots(2, 2)
    pl.subplots_adjust(hspace=.3)
    axs[0, 0].plot(x, np.array(expu_u)[:, 0], label='u')
    axs[0, 0].plot(x, np.array(expu_r)[:, 0], label='r')
    axs[0, 0].plot(x, np.array(expu_b)[:, 0], label='b')
    axs[0, 0].plot(x, np.array(expu_l)[:, 0], label='l')
    axs[0, 0].set_title('State 24 (Soph.)')
    axs[0, 0].legend()
    axs[0, 1].plot(x, np.array(expu_u)[:, 1], label='u')
    axs[0, 1].plot(x, np.array(expu_r)[:, 1], label='r')
    axs[0, 1].plot(x, np.array(expu_b)[:, 1], label='b')
    axs[0, 1].plot(x, np.array(expu_l)[:, 1], label='l')
    axs[0, 1].set_title('State 25 (Soph.)')
    axs[0, 1].legend()
    axs[1, 0].plot(x, np.array(expu_u)[:, 2], label='u')
    axs[1, 0].plot(x, np.array(expu_r)[:, 2], label='r')
    axs[1, 0].plot(x, np.array(expu_b)[:, 2], label='b')
    axs[1, 0].plot(x, np.array(expu_l)[:, 2], label='l')
    axs[1, 0].set_title('State 31 (Soph.)')
    axs[1, 0].legend()
    axs[1, 1].plot(x, np.array(expu_u)[:, 3], label='u')
    axs[1, 1].plot(x, np.array(expu_r)[:, 3], label='r')
    axs[1, 1].plot(x, np.array(expu_b)[:, 3], label='b')
    axs[1, 1].plot(x, np.array(expu_l)[:, 3], label='l')
    axs[1, 1].set_title('State 32 (Soph.)')
    axs[1, 1].legend()
    if isSoftmax:
        fig.suptitle('Forward: Using Softmax' + ' alpha: ' + str(alpha))
    else:
        fig.suptitle('Forward: \u03B5-greedy' + ' (\u03B5=' + str(epsilon)+')')
    fig.show()

else:
    x = [i for i in range(1, 1 + len(expu_u))]
    # first pic
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x, final_q_u[:, 0] - final_q_r[:, 0])
    axs[0].set_title('Difference Q(21, u) - Q(21, r) (Soph.)')
    axs[1].plot(x, final_q_l[:, 2] - final_q_u[:, 2])
    axs[1].set_title('Difference Q(9, l) - Q(9, u) (Soph.)')

    if isSoftmax:
        fig.suptitle('Forward: Using Softmax' + ' alpha: ' + str(alpha))
    else:
        fig.suptitle('Forward: \u03B5-greedy' + ' (\u03B5=' + str(epsilon)+')')
    fig.show()

    # second pic
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x, final_q_u[:, 0], label='u')
    axs[0].fill_between(x, final_q_u[:, 0] - final_q_u[:, 1], final_q_u[:, 0] + final_q_u[:, 1], alpha=0.2)
    print("(21, up)", np.array(expu_u)[:, 0][-1])
    axs[0].plot(x, final_q_r[:, 0], label='r')
    axs[0].fill_between(x, final_q_r[:, 0] - final_q_r[:, 1], final_q_r[:, 0] + final_q_r[:, 1], alpha=0.2)
    print("(21, right)", np.array(expu_r)[:, 0][-1])
    axs[0].plot(x, final_q_b[:, 0], label='b')
    axs[0].fill_between(x, final_q_b[:, 0] - final_q_b[:, 1], final_q_b[:, 0] + final_q_b[:, 1], alpha=0.2)
    print("(21, below)", np.array(expu_b)[:, 0][-1])
    axs[0].plot(x, final_q_l[:, 0], label='l')
    axs[0].fill_between(x, final_q_l[:, 0] - final_q_l[:, 1], final_q_l[:, 0] + final_q_l[:, 1], alpha=0.2)
    print("(21, left)", np.array(expu_l)[:, 0][-1])
    axs[0].set_title('Q(s=21) Soph.: Gridworld')
    axs[0].legend()
    axs[1].plot(x, final_q_u[:, 2], label='u')
    axs[1].fill_between(x, final_q_u[:, 2] - final_q_u[:, 3], final_q_u[:, 2] + final_q_u[:, 3], alpha=0.2)
    axs[1].plot(x, final_q_r[:, 2], label='r')
    axs[1].fill_between(x, final_q_r[:, 2] - final_q_r[:, 3], final_q_r[:, 2] + final_q_r[:, 3], alpha=0.2)
    axs[1].plot(x, final_q_b[:, 2], label='b')
    axs[1].fill_between(x, final_q_b[:, 2] - final_q_b[:, 3], final_q_b[:, 2] + final_q_b[:, 3], alpha=0.2)
    axs[1].plot(x, final_q_l[:, 2], label='l')
    axs[1].fill_between(x, final_q_l[:, 2] - final_q_l[:, 3], final_q_l[:, 2] + final_q_l[:, 3], alpha=0.2)
    axs[1].set_title('Q(s=9) Soph.: Gridworld')
    plt.legend()
    if isSoftmax:
        fig.suptitle('Forward: using Softmax' + ' alpha: ' + str(alpha))
    else:
        fig.suptitle('Forward: \u03B5-greedy' + ' (\u03B5=' + str(epsilon)+')')
    fig.show()





