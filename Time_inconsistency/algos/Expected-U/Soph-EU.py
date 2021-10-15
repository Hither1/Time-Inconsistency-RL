"""
Created on Mon Feb 2

@author: Huangyuan

Policy ---> agent in original paper

"""

from envs.DoughVeg_gridworld import GridworldEnv
#from envs.DoughVeg_windy import GridworldEnv
#from envs.DoughVeg_simple_stochastic import GridworldEnv # stochastic simple gridworld
import numpy as np
import pandas as pd
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.special import softmax
import time
import pylab as pl
import seaborn as sns
sns.set_style("whitegrid")

current_env_windy = False
#current_env_windy = True  # Change between normal/windy gridworlds
isSoftmax = False

discount_factor = 1
reward_multiplier = 1
discounting = 'hyper'  # 'hyper', 'exp'
init_policy = 'random'  # 'random' 'stable'

# The noise parameter that modulates between random choice (=0) and perfect maximization (=\infty)

num_episodes = 100000

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
    alpha = .001  # The noise parameter that modulates between random choice (=0) and perfect maximization (=\infty)
else:
    epsilon = .07
step_size = .4
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
    global critical_episode_9, critical_episode_21, num_bad_episode
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
            for d in range(30):
                expUtility[8][d][a_] = 10 * discount(d) * reward_multiplier
                expUtility[2][d][a_] = 19 * discount(d) * reward_multiplier


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


            probs = agent(state)  # Select an action according to policy

            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)

            #while state == next_state:
            #    action = np.random.choice(np.arange(len(probs)), p=probs)
            #    next_state, reward, done, _ = env.step(action)

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


            else:
                expectation = np.dot(expUtility[next_state][t + 1], agent(next_state).T)


            expUtility[state][t][action] = (1 - step_size) * expUtility[state][t][action] + step_size * (u + expectation)
            #Utility

            if done:
                episode.append((next_state, 0, 0))
                break
            state = next_state  # update to the next state

            if state == 9 and action == 0:  # check for possible JUMP at 9 due to noisy policy

                if critical_episode_9 == 0:
                    critical_episode_9 = i_episode - 1



        if next_state != 2 and next_state != 8:
            num_bad_episode += 1

        if current_env_windy:
            # Track Q[24] for all actions and plot
            expu_u.append([expUtility[24][0][0], expUtility[25][0][0], expUtility[31][0][0], expUtility[32][0][0]])
            expu_r.append([expUtility[24][0][1], expUtility[25][0][1], expUtility[31][0][1], expUtility[32][0][1]])
            expu_b.append([expUtility[24][0][2], expUtility[25][0][2], expUtility[31][0][2], expUtility[32][0][2]])
            expu_l.append([expUtility[24][0][3], expUtility[25][0][3], expUtility[31][0][3], expUtility[32][0][3]])
        else:
            # Track Q[21] for all actions and plot
            expu_u.append([expUtility[x][0][0] for x in range(24)])
            expu_r.append([expUtility[x][0][1] for x in range(24)])
            expu_b.append([expUtility[x][0][2] for x in range(24)])
            expu_l.append([expUtility[x][0][3] for x in range(24)])
            V.append([max(expUtility[21][0])])


    return expUtility


# Critical to check how sensitive is relevant states (i.e. 13, 17, 21) to sudden deviation at 9
critical_states_update_order = []
critical_index_9 = 0
critical_episode_9 = 0
critical_index_21 = 0
critical_episode_21 = 0

np.random.seed(0)
V_s, q_u_s, q_r_s, q_b_s, q_l_s = [], [], [], [], []
num_bad_episodes = []
for _ in range(50):
    expu_correction_21 = []
    V, expu_u, expu_r, expu_b, expu_l = [], [], [], [], []
    num_bad_episode = 0
    expUtility = td_control(env, num_episodes, isSoftmax=isSoftmax, step_size=step_size)
    V, expu_u, expu_r, expu_b, expu_l = np.array(V), np.array(expu_u), np.array(expu_r), np.array(expu_b), np.array(expu_l)


    q_u_s.append(expu_u)
    q_r_s.append(expu_r)
    q_b_s.append(expu_b)
    q_l_s.append(expu_l)
    V_s.append(V)
    num_bad_episodes.append(num_bad_episode)


V_s, q_u_s, q_r_s, q_b_s, q_l_s = np.array(V_s), np.array(q_u_s), np.array(q_r_s), np.array(q_b_s), np.array(q_l_s)

final_q_u, final_q_r, final_q_b, final_q_l, final_V = [], [], [], [], []
for i in range(num_episodes):
    final_V.append([np.mean(V_s[:, i]), np.std(V_s[:, i])])
    final_q_u.append([np.mean(q_u_s[:, i, x]) for x in range(24)] + [np.std(q_u_s[:, i, x]) for x in range(24)])
    final_q_r.append([np.mean(q_r_s[:, i, x]) for x in range(24)] + [np.std(q_r_s[:, i, x]) for x in range(24)])
    final_q_b.append([np.mean(q_b_s[:, i, x]) for x in range(24)] + [np.std(q_b_s[:, i, x]) for x in range(24)])
    final_q_l.append([np.mean(q_l_s[:, i, x]) for x in range(24)] + [np.std(q_l_s[:, i, x]) for x in range(24)])
final_V, final_q_u, final_q_r, final_q_b, final_q_l = np.array(final_V), np.array(final_q_u), np.array(
    final_q_r), np.array(final_q_b), np.array(final_q_l)

df_V, df_u, df_r, df_b, df_l = pd.DataFrame(final_V), pd.DataFrame(final_q_u), pd.DataFrame(final_q_r), pd.DataFrame(final_q_b), pd.DataFrame(final_q_l)
df_V.to_csv("../../results/simple/fwd/round_9/V_values_" + str(step_size) + ".csv")
df_u.to_csv("../../results/simple/fwd/round_9/Q_values_" + str(step_size)  + "/u.csv")
df_r.to_csv("../../results/simple/fwd/round_9/Q_values_" + str(step_size)  + "/r.csv")
df_b.to_csv("../../results/simple/fwd/round_9/Q_values_" + str(step_size)  + "/b.csv")
df_l.to_csv("../../results/simple/fwd/round_9/Q_values_" + str(step_size)  + "/l.csv")

# ------------------------------------------------------------------------------------------------

'''Checking criticals'''
i_21, i_9 = 0, 0
for i in range(num_episodes - 1, 0, -1):
    if final_q_r[i][21] >= final_q_u[i][21] and final_q_r[i - 1][21] < final_q_u[i - 1][21]:
        print("first overtake episode (average) of s=21 is ", i)
        i_21 = i
        break
for i in range(num_episodes - 1, -1, -1):
    if final_q_l[i][9] >= final_q_u[i][9] and final_q_l[i - 1][9] < final_q_u[i - 1][9]:
        print("first overtake episode (average) of s=9 is ", i)
        i_9 = i
        break

print("average number of bad episodes", sum(num_bad_episodes)/len(num_bad_episodes))
print("Final Best Actions at 20K:")
nr = env.shape[0]
nc = env.shape[1]
'''
#print("Average number of bad episodes ", sum(num_bad_episodes) / len(num_bad_episodes))
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

    plt.plot(x, final_V[:], label='u')
    plt.fill_between(x, final_V[:, 0] - final_V[:, 1], final_V[:, 0] + final_V[:, 1], alpha=0.2)
    print("(21, up)", np.array(expu_u)[:, 0][-1])
    print("(21, right)", np.array(expu_r)[:, 0][-1])
    print("(21, below)", np.array(expu_b)[:, 0][-1])
    print("(21, left)", np.array(expu_l)[:, 0][-1])
    plt.title('V Soph.: Gridworld')
    x_range = np.arange(0, num_episodes, step=int(num_episodes/5))
    y_range = [final_V[x, 0] for x in x_range]
    y_err = [final_V[x, 1] for x in x_range]
    plt.legend()
    if isSoftmax:
        plt.suptitle('Forward: using Softmax' + ' alpha: ' + str(alpha) + ' step_size: ' + str(step_size))
    else:
        plt.suptitle('Forward: \u03B5-greedy' + ' (\u03B5=' + str(epsilon)+')' + ' step_size: ' + str(step_size))
    plt.show()

'''
for r in [19999, 29999, 99999]:
    nr = env.shape[0]
    nc = env.shape[1]
    for i in []:
        print("Final Best Actions at: " + str(r+1))
    for r_ in range(nr):
        row = []
        for c_ in range(nc):
            row.append(np.argmax([final_q_u[r][r_ * nc + c_],
                                  final_q_r[r][r_ * nc + c_],
                                  final_q_b[r][r_ * nc + c_],
                                  final_q_l[r][r_ * nc + c_]]
                                 ))
        print(row)

    print("/")
    x = [i for i in range(1 + r)]
    linewidth = 3
    plt.ylim([0, 4])
    plt.xlim([0, r])
    plt.plot(x, final_q_u[:r+1, 21], label='UP', linewidth=linewidth)
    plt.fill_between(x, final_q_u[:r+1, 21] - final_q_u[:r+1, 21+24], final_q_u[:r+1, 21] + final_q_u[:r+1, 21+24], alpha=0.2)

    plt.plot(x, final_q_r[:r+1, 21], label='RIGHT', linewidth=linewidth)
    plt.fill_between(x, final_q_r[:r+1, 21] - final_q_r[:r+1, 21+24], final_q_r[:r+1, 21] + final_q_r[:r+1, 21+24], alpha=0.2)

    plt.title('Q(s=21) SophEU '+'(\u03B5 = .07,\u03B1' + r'$_{Q}$' +'=.4,' + r"$\bar{T} = 100$)" + " Deterministic")
    plt.legend()
    plt.annotate("Overtake at " + str(i_21), (i_21, final_q_r[i_21][21] - 0.3))
    plt.show()

    plt.xlim([0, r])
    plt.ylim([0, 6.1])
    plt.plot(x, final_q_u[:r + 1, 9], label='UP', linewidth=linewidth)
    plt.fill_between(x, final_q_u[:r + 1, 9] - final_q_u[:r + 1, 9 + 24],
                     final_q_u[:r + 1, 9] + final_q_u[:r + 1, 9 + 24], alpha=0.2)

    plt.plot(x, final_q_l[:r + 1, 9], label='LEFT', color="green", linewidth=linewidth)
    plt.fill_between(x, final_q_l[:r + 1, 9] - final_q_l[:r + 1, 9 + 24],
                     final_q_l[:r + 1, 9] + final_q_l[:r + 1, 9 + 24], color="green", alpha=0.2)
    plt.title(
        'Q(s=9) SophEU ' + '(\u03B5 = .07,\u03B1' + r'$_{Q}$' + '=.4,' + r"$\bar{T} = 100$)" + " Deterministic")
    plt.annotate("Overtake at " + str(i_9), (i_9, final_q_u[i_9][9] - 0.3))
    plt.legend()
    plt.show()

