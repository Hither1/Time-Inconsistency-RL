from envs.DoughVeg_gridworld import GridworldEnv
#  from envs.DoughVeg_windy import GridworldEnv
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

current_env_windy = False  #

discount_factor = 1
bias = 0
sigma = 1
discounting = 'hyper'  # 'hyper', 'exp'
init_policy = 'random'  # 'random' 'stable'

epsilon = .6
num_episodes = 30000  # 0000

env = GridworldEnv()

is_wall = lambda s: s in [6, 10, 14, 18]


# is_wall = lambda s: s in [6, 13, 20, 27, 35, 36, 37, 38, 39]  # windy gridworld

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


# Q-learning (off-policy TD control)
def td_control(env, num_episodes, step_size):
    if current_env_windy:
        global Q_24_u, Q_24_r, Q_24_b, Q_24_l
    else:
        global Q_correction_21, Q_u, Q_r, Q_b, Q_l

    # The type of discounting
    # discount = auto_discounting()

    # Global variables
    global critical_episode_9, critical_episode_21

    # Initialize the matrix with Q-values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    break_out = 0

    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    for i_episode in range(1, num_episodes + 1):

        if break_out == 1:
            break

        print(i_episode)

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
            # if-else condition only for windy gridworld
            '''if state == 31:  # the blue moon
                action = 0  # go up
            else:
                action = np.random.choice(np.arange(len(probs)), p=probs)'''
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))

            # Update q_value for a state-action pair Q(s,a):
            # Q(s,a) = Q(s,a) + α( r + γmaxa' Q(s',a') - Q(s,a) )
            state, action, reward = episode[t]

            # Do the computation

            r = reward if reward else 1
            delta = reward - Q[state][action] + max(Q[next_state]) - discount_factor * Q[state][action] * max(
                Q[next_state])/(bias + r)**sigma  # delta is the time-difference error

            if delta > 1e+2:  # != 0:
                print("Delta ", delta)
                print('reward:', reward)
                print('state:', state)
                print('action:', action)
                print('Q[s, a]:', Q[state][action])
                print('next_state:', next_state)
                print('Q[next_state]:', Q[next_state])
                break_out = 1
                break

            Q[state][action] += step_size * delta

            if done:
                break
            state = next_state

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
        if current_env_windy:
            # Track Q[24] for all actions and plot
            Q_24_u += [Q[24][0]]
            Q_24_r += [Q[24][1]]
            Q_24_b += [Q[24][2]]
            Q_24_l += [Q[24][3]]
        else:
            # Track Q[21] for all actions and plot
            Q_u.append([Q[21][0], Q[9][0]])
            Q_r.append([Q[21][1], Q[9][1]])
            Q_b.append([Q[21][2], Q[9][2]])
            Q_l.append([Q[21][3], Q[9][3]])
            print("Q_u", Q_u)

    return Q


# Initialize the arrays to store values of Q
if current_env_windy:
    Q_24_u = []
    Q_24_r = []
    Q_24_b = []
    Q_24_l = []
else:
    # Critical to check how sensitive is relevant states (i.e. 13, 17, 21) to sudden deviation at 9
    critical_states_update_order = []
    critical_index_9 = 0
    critical_episode_9 = 0
    critical_index_21 = 0
    critical_episode_21 = 0

    Q_correction_21 = []
    Q_u = []  # [value ar 21, value at 9] for action: up
    Q_r = []
    Q_b = []
    Q_l = []

np.random.seed(0)

Q = td_control(env, num_episodes, step_size=0.1)

# ------------------------------------------------------------------------------------------------

'''Checking criticals'''
print('EPSILON:', epsilon)
print('POLICY INIT:', init_policy)
print('DISCOUNTING:', discounting)
print('-----')
if current_env_windy:
    print()
else:
    print('(9, UP) first appears at ep:', critical_episode_9)
    print('visitation to 21 aft:', critical_states_update_order[critical_index_9:].count(21))
    # print('(21, RIGHT) first appears at ep:', critical_episode_21)
    print('Q(21, RIGHT) > Q(21, UP) first appears at ep:', critical_episode_21)

if current_env_windy:
    x = [i for i in range(1, 1 + len(Q_24_u))]
    # first pic
    plt.figure()
    plt.plot(x, np.array(Q_24_u) - np.array(Q_24_r))
    plt.title("Difference Q(24, u) - Q(24, r) (HDTD)")
    plt.show()

    # second pic
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(x, Q_24_u, label='u')
    axs[0, 0].plot(x, Q_24_r, label='r')
    axs[0, 0].plot(x, Q_24_b, label='b')
    axs[0, 0].plot(x, Q_24_l, label='l')
    axs[0, 0].set_title('State 24')
    axs[0, 1].plot(x, Q_24_u, label='u')
    axs[0, 1].plot(x, Q_24_u, label='u')
    axs[0, 1].plot(x, Q_24_u, label='u')
    axs[0, 1].plot(x, Q_24_u, label='u')
    axs[0, 1].set_title('State 25')
    axs[1, 0].plot(x, Q_24_u, label='u')
    axs[1, 0].plot(x, Q_24_u, label='u')
    axs[1, 0].plot(x, Q_24_u, label='u')
    axs[1, 0].plot(x, Q_24_u, label='u')
    axs[1, 0].set_title('State 31')
    axs[1, 1].plot(x, Q_24_u, label='u')
    axs[1, 1].plot(x, Q_24_u, label='u')
    axs[1, 1].plot(x, Q_24_u, label='u')
    axs[1, 1].plot(x, Q_24_u, label='u')
    axs[1, 1].set_title('State 32')

    plt.legend()
    plt.title("HDTD Complete: Windy Gridworld")
    plt.show()
else:
    x = [i for i in range(1, 1 + len(Q_u))]
    # first pic
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x, np.array(Q_u)[:, 0] - np.array(Q_r)[:, 0])
    axs[0].set_title('Difference Q(21, u) - Q(21, r) (HDTD Complete)')
    axs[1].plot(x, np.array(Q_l)[:, 1] - np.array(Q_u)[:, 1])
    axs[1].set_title('Difference Q(9, l) - Q(9, u) (HDTD Complete)')
    fig.show()

    # second pic
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(x, np.array(Q_u)[:, 0], label='u')
    axs[0].plot(x, np.array(Q_r)[:, 0], label='r')
    axs[0].plot(x, np.array(Q_b)[:, 0], label='b')
    axs[0].plot(x, np.array(Q_l)[:, 0], label='l')
    axs[0].set_title('Q(s=21) HDTD Complete: Gridworld')
    axs[1].plot(x, np.array(Q_u)[:, 1], label='u')
    axs[1].plot(x, np.array(Q_r)[:, 1], label='r')
    axs[1].plot(x, np.array(Q_b)[:, 1], label='b')
    axs[1].plot(x, np.array(Q_l)[:, 1], label='l')
    axs[1].set_title('Q(s=9) HDTD Complete: Gridworld')
    plt.legend()
    fig.show()
