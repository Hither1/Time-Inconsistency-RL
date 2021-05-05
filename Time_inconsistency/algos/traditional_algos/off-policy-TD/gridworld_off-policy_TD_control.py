from envs.DoughVeg_gridworld import GridworldEnv
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

discount_factor = 0.2
discounting = 'hyper'  # 'hyper', 'exp'
init_policy = 'random'  # 'random' 'stable'

epsilon = .1
num_episodes = 70000  # 0000

env = GridworldEnv()

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


'''
def expUtility(state, action, delay):
    if():
        return 0
    else:
        u = 1/(1 + discount_factor * delay) * Q(state, action)
        next_state, reward, done, _ = env.step(action)
        next_action =
        return u + expUtility(next_state, next_action, delay+1)
'''


# Q-learning (off-policy TD control)
def td_control(env, num_episodes, step_size):
    global Q_correction_21, Q_21_u, Q_21_r, Q_21_b, Q_21_l

    # The type of discounting
    discount = auto_discounting()

    # Global variables
    global critical_episode_9, critical_episode_21

    # Initialize the matrix with Q-values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    for i_episode in range(1, num_episodes + 1):
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
            probs = policy(state)
            print("current state " + str(state))
            print("probs" + str(probs))
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state

        for i in range(len(episode)):
            state_i, action_i, reward_i = episode[i]
            Q[state_i][action_i] = 0
            for j in range(i, len(episode)):  # add up the discounted future rewards
                state_j, action_j, reward_j = episode[j]
                print("discount value is " + str(discount(j - i)))
                Q[state_i][action_i] += discount(j - i) * reward_j

        for t in range(len(episode)):
            # Update q_value for a state-action pair Q(s,a):
            # Q(s,a) = Q(s,a) + α( r + γmaxa' Q(s',a') - Q(s,a) )
            state, action, reward = episode[t]
            # get the max of q value of next state
            if t == len(episode) - 1:
                break
            next_state, next_action, next_reward = episode[t + 1]  # the next state that happened during the episode
            max_q_sa_next = max(Q[next_state])
            # Do the computation
            Q[state][action] = Q[state][action] + step_size * (max_q_sa_next - Q[state][action])

            if state == 9 and action == 0:  # check for possible JUMP at 9 due to noisy policy

                if critical_episode_9 == 0:
                    critical_episode_9 = i_episode - 1

                # count_9 += 1

                curr_policy = np.argmax(policy(state))
                curr_Q = Q[state]
                print('curr_policy_9:', curr_policy)
                print('curr_Q[9]:', curr_Q)

                print('-----')

            # if state == 21 and action == 1: # check for when the change to SPE is reflected at 21
            if state == 21 and Q[21][1] > Q[21][0]:

                if critical_episode_21 == 0:  # update the episode in which we reach state 21
                    critical_episode_21 = i_episode - 1

                # count_21 += 1

                curr_policy = np.argmax(policy(state))
                curr_Q = Q[state]
                print('curr_policy:', curr_policy)
                print('curr_Q[21]:', curr_Q)

                # print('trajectory aft 21:', episode[first_occurence_idx:])

                print('--------------')

        # Track Q[21] for all actions and plot
        Q_21_u += [Q[21][0]]
        Q_21_r += [Q[21][1]]
        Q_21_b += [Q[21][2]]
        Q_21_l += [Q[21][3]]
    return Q


# Critical to check how sensitive is relevant states (i.e. 13, 17, 21) to sudden deviation at 9
critical_states_update_order = []
critical_index_9 = 0
critical_episode_9 = 0
critical_index_21 = 0
critical_episode_21 = 0

Q_correction_21 = []
Q_21_u = []
Q_21_r = []
Q_21_b = []
Q_21_l = []
np.random.seed(0)

Q = td_control(env, num_episodes, step_size=0.5)

# ------------------------------------------------------------------------------------------------

'''Checking criticals'''
print('EPSILON:', epsilon)
print('POLICY INIT:', init_policy)
print('DISCOUNTING:', discounting)
print('-----')
print('(9, UP) first appears at ep:', critical_episode_9)
print('visitation to 21 aft:', critical_states_update_order[critical_index_9:].count(21))
# print('(21, RIGHT) first appears at ep:', critical_episode_21)
print('Q(21, RIGHT) > Q(21, UP) first appears at ep:', critical_episode_21)

x = [i for i in range(1, 1 + len(Q_21_u))]

plt.figure()
plt.plot(x, np.array(Q_21_u) - np.array(Q_21_r))
plt.show()

plt.figure()
plt.plot(x, Q_21_u, label='u')
plt.plot(x, Q_21_r, label='r')
plt.plot(x, Q_21_b, label='b')
plt.plot(x, Q_21_l, label='l')
plt.legend()
plt.title("Off-policy TD Control")
plt.show()
