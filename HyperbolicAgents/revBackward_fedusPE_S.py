import numpy as np
import pandas as pd
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import pylab as pl
from scipy.special import softmax
from datetime import datetime

import sys
import os

cwd = os.getcwd()
sys.path.append(cwd) # nb: append cwd or "../" not enough, need to add __init__ to the cwd;
if "../" not in sys.path:
    sys.path.append("../")

current_env_windy = False #True
env_ID = 'S' # 'S'
run_ID = datetime.now().strftime("%d%m%Y%H%M%S") # num_runs >= 10

G_seed = 0 # for sweeping: G_low, G_high, G_distrib, G_seed, G_size
G_low, G_high, G_distrib = 0.01, 0.99, 'Unif'

if env_ID == 'D':   
    from lib.envs.Gridworld_D import GridworldEnv # stochastic simple gridworld
    step_size_1 = .4
    step_size_2 = 1.0
    
    np.random.seed(G_seed)
    G_size = 1000 # 1000 better, 200 for fair comparison on the num of vf
    G = np.sort(np.random.uniform(G_low, G_high, G_size))
else:
    from lib.envs.Gridworld_S import GridworldEnv
    step_size_1 = .4
    step_size_2 = .9
    
    np.random.seed(G_seed)
    G_size = 1000
    G = np.sort(np.random.uniform(G_low, G_high, G_size))

algo_ID = 'fwdFedus' # 'MC', 'bwd-rev', 'sophEU', 'Fedus'
if algo_ID == 'sophEU':
    num_episodes = 100000  #30k - 100k
else:
    num_episodes = 30000
num_seeds = 50

discount_factor = 1 # "k" in \varphi(t) = 1 / (1 + k.t)
reward_multiplier = 1
episode_length = 100 # 1000
discounting = 'hyper'  # 'hyper', 'exp'

hyperparams = [[step_size_1, step_size_2], [episode_length], [G_seed, G_size, G_low, G_high, G_distrib], G]

isSoftmax = False
if isSoftmax:
    alpha = 3  # The noise parameter that modulates between random choice (=0) and perfect maximization (=\infty)
    hyperparams = [[alpha]] + hyperparams
else:
    epsilon = .07
    hyperparams = [[epsilon]] + hyperparams

################################### SAVELOAD ##################################
import csv

# initialize docs: .csv -> Q_values, .txt -> hyperparams (.txt shd be equal across runID)
title = "{}_{}_{}".format(algo_ID, env_ID, run_ID)

df0 = pd.DataFrame(['Seed', 'Vf', 'state', 'action'] + [None for i in range(num_episodes)])
df0.to_csv('results/' + title + '.csv', header = None, index = False)

f = open('results/' + title + '.txt', 'a', newline = '')
writer = csv.writer(f)
writer.writerows(hyperparams)
f.close()

############################### RUNNER ########################################

# Init Env
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

def choose_color(number):
    '''
    For the purpose of choosing color dynamically so that we can have multiple colors in one line
    '''
    if number == 0:
        return 'b'
    elif number == 1:
        return 'r'
    elif number == 2:
        return 'g'
    else:
        return 'r'

def make_policy(Q, nA, isSoftmax):
    
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)

        return A

    def policy_fn_softmax(observation):
        return softmax(Q[observation] * alpha)

    if isSoftmax:
        return policy_fn_softmax
    else:
        return policy_fn


# Hyperbolic Discounted Q-learning (off-policy TD control)
def td_control(env, num_episodes, isSoftmax, step_size_1, step_size_2, G):
    
    global q_correction_21, q_u, q_r, q_b, q_l, num_bad_episode
    global critical_episode_21, revisits, Q_episodes

    discount = auto_discounting()

    # Init Q-dict, Q_g-dict
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    if current_env_windy:  # windy gridworld -
        for a in range(env.action_space.n):
            Q[10][a] = 3 * reward_multiplier
            Q[28][a] = 10 * reward_multiplier
            Q[4][a] = 6 * reward_multiplier
    else:
        for a in range(env.action_space.n):
            Q[2][a] = 19 * reward_multiplier
            Q[8][a] = 10 * reward_multiplier
    
    Q_g = defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n)))
    for g in G:
        for a in range(env.action_space.n):
            Q_g[g][2][a] = 19 * reward_multiplier
            Q_g[g][8][a] = 10 * reward_multiplier
    
    
    # Take the Utility function from reward function of the environment
    # Utility = env.copy_reward_fn
    agent = make_policy(Q, env.action_space.n, isSoftmax=isSoftmax) # randomized policy, w/o 'policy-table'
    policy = defaultdict(lambda: 0)

    first_time_right_larger = False
    
    #f: [t][m][s][a] t is current time, m is time at which the reward is received
    f = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(env.action_space.n))))
        
    for i_episode in range(1, num_episodes + 1):
        states = set({})  # keep a set of states we already visited

        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes))
            sys.stdout.flush()

        # Generate an episode: array of (state, action, reward) tuples
        episode = []
        
        state = env.reset()
        while is_wall(state) or state == 2 or state == 8:
            state = env.reset()

        current_revisit = 0
        
        for t in range(episode_length):

            if state in states:
                current_revisit += 1
            else:
                states.add(state)

            probs = agent(state)  
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            
            # the above 'reward' is r(next_state)

            if next_state != state:
                episode.append((state, action, reward))
            if done:
                if current_env_windy:
                    if next_state == 4:
                        episode.append((next_state, np.argmax(Q[state]), 6 * reward_multiplier))
                    elif next_state == 10:
                        episode.append((next_state, np.argmax(Q[state]), 3 * reward_multiplier))
                    elif next_state == 28:
                        episode.append((next_state, np.argmax(Q[state]), 10 * reward_multiplier))
                else:
                    if next_state == 2:
                        episode.append((next_state, np.argmax(Q[state]), 19 * reward_multiplier))
                    elif next_state == 8:
                        episode.append((next_state, np.argmax(Q[state]), 10 * reward_multiplier))
                break
            
            state = next_state  # update to the next state
            
        revisits.append(current_revisit)

        # Offline update, backward
        if current_env_windy:
            for a in range(env.action_space.n):
                if next_state == 4:
                    f[len(episode) - 1][4][a] = 5
                    f[len(episode) - 1][10][a] = 0
                    f[len(episode) - 1][28][a] = 0
                elif next_state == 10:
                    f[len(episode) - 1][4][a] = 0
                    f[len(episode) - 1][10][a] = 3
                    f[len(episode) - 1][28][a] = 0
                elif next_state == 28:
                    f[len(episode) - 1][4][a] = 0
                    f[len(episode) - 1][10][a] = 0
                    f[len(episode) - 1][28][a] = 10
        else:
            for a in range(env.action_space.n):
                if next_state == 2:
                    f[len(episode) - 1][len(episode) - 1][2][a] = 19 * reward_multiplier
                    # f[len(episode) - 1][len(episode) - 1][8][a] = 0
                elif next_state == 8:
                    # f[len(episode) - 1][len(episode) - 1][2][a] = 0
                    f[len(episode) - 1][len(episode) - 1][8][a] = 10 * reward_multiplier

        if next_state != 2 and next_state != 8:
            num_bad_episode += 1
        
        # 2. others, t = T-2: 0:
        for t in range(len(episode)-1): #range(len(episode) - 2, -1, -1):
            # Initialize the boundary values for f

            s, a, r = episode[t] # use prev_r, not this r! the reason 'bwd' works
            # is cause it does not use the 'r' at all!
            next_state, next_action, next_r = episode[t + 1]
            
            '''
            for m in range(t+1, len(episode)):
                f_value = f[t + 1][m][next_state][policy[next_state]] / discount(m - (t + 1)) * discount(m - t)
                f[t][m][s][a] = f[t][m][s][a] + step_size_2 * (f_value - f[t][m][s][a])

            Q[s][a] = Q[s][a] + step_size_1 * (Q[next_state][policy[next_state]] -
                    max(0, sum([f[t + 1][m][next_state][policy[next_state]] - f[t][m][s][a] for m in range(t+1, len(episode))]))
                                               - Q[s][a])
            '''
            
            # Init Q_g w/ terminal values
            for g in G:
                '''
                if s == 1:
                    print('g:', g)
                    print('a:', a)
                    print('next_s:', next_state)
                    print('next_a:', policy[next_state])
                    print('Q_g, bf:', Q_g[g][s][a])
                    print('--')'''
                
                # in the following, r = 0 since we have boundary-only rewards
                Q_g[g][s][a] += step_size_1 * (0 + g * Q_g[g][next_state][policy[next_state]] - Q_g[g][s][a])
                '''
                if s == 1:
                    print('target:', 0 + g * Q_g[g][next_state][policy[next_state]])
                    print('r:', 0)
                    print('undiscounted next Q_g:',  Q_g[g][next_state][policy[next_state]])
                    print('--')
                    print('Q_g, aft:', Q_g[g][s][a])
                    print('===')
                    
                    if r == 19 and Q_g[g][next_state][policy[next_state]] == 19:
                        print('episode:', episode)
                        print('t:', t)
                        raise ValueError()
                    
                    # Target for some g* are sometimes too large, e.g.  26. ..
                    # this is caused by Q[g*][5][a[5]] also too large! e.g. 28. ...
                '''    
            
            Q[s][a] = np.sum([(G[i+1] - G[i]) * Q_g[G[i]][s][a] for i in range(len(G)-1)])
            
            policy[s] = np.argmax(Q[s]) # add 'if-conditions'

        if current_env_windy:
            # Track Q[24] for all actions and plot
            q_u.append([Q[24][0], Q[25][0], Q[31][0], Q[32][0]])
            q_r.append([Q[24][1], Q[25][1], Q[31][1], Q[32][1]])
            q_b.append([Q[24][2], Q[25][2], Q[31][2], Q[32][2]])
            q_l.append([Q[24][3], Q[25][3], Q[31][3], Q[32][3]])
        else:
            if Q[21][1] > Q[21][3] and Q[21][1] > Q[21][0] and not first_time_right_larger:
                critical_episode_21 = i_episode
                first_time_right_larger = True
            
            '''
            print("episode", i_episode)
            print(Q[21])
            print(Q[9])
            if Q[21][0] > 2.08:
                print("Checking ")
                print(episode)
            '''
            
            Q_episodes.append(episode)
            Q_episodes.append(list(Q[21]))

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

q_u_s = []
q_r_s = []
q_b_s = []
q_l_s = []
num_bad_episodes = []

for seed_ in range(num_seeds):
    revisits = []
    q_correction_21 = []
    q_u = []
    q_r = []
    q_b = []
    q_l = []
    num_bad_episode = 0
    Q_episodes = []
    
    np.random.seed(0)
    
    Q = td_control(env, num_episodes, isSoftmax, 
                   step_size_1=step_size_1, step_size_2=step_size_2, G = G)
    
    # update docs
    df = pd.read_csv('results/' + title + '.csv', header = None, dtype = object)
    
    q_u = np.array(q_u)
    q_r = np.array(q_r)
    q_b = np.array(q_b)
    q_l = np.array(q_l)
    q = np.concatenate((np.array([q_u[:, 0]]).T, np.array([q_r[:, 0]]).T, 
                        np.array([q_b[:, 0]]).T, np.array([q_l[:, 0]]).T,
                        np.array([q_u[:, 1]]).T, np.array([q_r[:, 1]]).T, 
                        np.array([q_b[:, 1]]).T, np.array([q_l[:, 1]]).T), axis = 1)
    
    idx = 0
    prev_idx = 0
    #to_append = np.array([])
    for state in [21, 9]:
        for vf in ['Q', 'pi', 'V']:
            if vf == 'Q':
                for action in range(4):
                    #to_append = np.concatenate((to_append, np.array([seed, vf, state, action]).T),
                    #                           axis = 1)
                    #df[seed_] = [vf, state, action] + list(q[:, idx])
                    
                    to_append = pd.DataFrame([seed_, vf, state, action] + list(q[:, idx]))
                    df = pd.concat([df, to_append], axis = 1)
                    
                    idx += 1
                    
            elif vf == 'pi':
                print('prev idx:', prev_idx)
                print('idx:', idx) # 4 -> 8
                pi_idx = np.argmax(q[:, prev_idx:idx], axis = 1)
                to_append = pd.DataFrame([seed_, vf, state, None] + list(pi_idx))
                df = pd.concat([df, to_append], axis = 1)
                
            else:
                to_append = pd.DataFrame([seed_, vf, state, None] + [q[i, pi_idx[i] + prev_idx] for i in range(len(pi_idx))])
                # pi_idx \in [0, 1, 2, 3]
                df = pd.concat([df, to_append], axis = 1)
                prev_idx = idx
    
    df.to_csv('results/' + title + '.csv', header = None, index = False)
    
    ################
    df = pd.DataFrame(Q_episodes)
    df.to_csv('results/' + title + '_Q21_n_episode_{}.csv'.format(seed_))
    
    q_u_s.append(q_u)
    q_r_s.append(q_r)
    q_b_s.append(q_b)
    q_l_s.append(q_l)
    num_bad_episodes.append(num_bad_episode)


######################### AVG + STDEV COMPUTE #################################

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
        row.append(np.argmax(Q[r_ * nc + c_]))
    print(row)

if current_env_windy:
    print("Final Best Actions with Wind: ")
    for r_ in range(nr):
        row = []
        for c_ in range(nc):
            if r_ * nc + c_ == 24 or r_ * nc + c_ == 31:
                a = 0 if Q[r_ * nc + c_][0] > Q[r_ * nc + c_][1] else 1
                row.append(a)
            else:
                row.append(np.argmax(Q[r_ * nc + c_]))
        print(row)

    x = [i for i in range(1, 1 + len(q_u))]
    # first pic
    fig, axs = plt.subplots(2, 2)
    pl.subplots_adjust(hspace=.3)
    axs[0, 0].plot(x, np.array(np.array(q_u)[:, 0]) - np.array(np.array(q_r)[:, 0]))
    axs[0, 0].set_title('Q(24, u) - Q(24, r)')
    axs[0, 1].plot(x, np.array(np.array(q_u)[:, 1]) - np.array(np.array(q_b)[:, 1]))
    axs[0, 1].set_title('Q(25, u) - Q(25, b)')
    axs[1, 0].plot(x, np.array(np.array(q_l)[:, 2]) - np.array(np.array(q_u)[:, 2]))
    axs[1, 0].set_title('Q(31, l) - Q(31, u)')
    axs[1, 1].plot(x, np.array(np.array(q_u)[:, 3]) - np.array(np.array(q_l)[:, 3]))
    axs[1, 1].set_title('Q(32, u) - Q(32, l)')
    if isSoftmax:
        fig.suptitle('Backward: Using Softmax' + ' alpha: ' + str(alpha))
    else:
        fig.suptitle('Backward: \u03B5-greedy' + ' (\u03B5=' + str(epsilon)+')')
    fig.show()

    # second pic
    fig, axs = plt.subplots(2, 2)
    pl.subplots_adjust(hspace=.3)
    axs[0, 0].plot(x, np.array(q_u)[:, 0], label='u')
    axs[0, 0].plot(x, np.array(q_r)[:, 0], label='r')
    axs[0, 0].plot(x, np.array(q_b)[:, 0], label='b')
    axs[0, 0].plot(x, np.array(q_l)[:, 0], label='l')
    axs[0, 0].set_title('State 24')
    axs[0, 0].legend()
    axs[0, 1].plot(x, np.array(q_u)[:, 1], label='u')
    axs[0, 1].plot(x, np.array(q_r)[:, 1], label='r')
    axs[0, 1].plot(x, np.array(q_b)[:, 1], label='b')
    axs[0, 1].plot(x, np.array(q_l)[:, 1], label='l')
    axs[0, 1].set_title('State 25')
    axs[0, 1].legend()
    axs[1, 0].plot(x, np.array(q_u)[:, 2], label='u')
    axs[1, 0].plot(x, np.array(q_r)[:, 2], label='r')
    axs[1, 0].plot(x, np.array(q_b)[:, 2], label='b')
    axs[1, 0].plot(x, np.array(q_l)[:, 2], label='l')
    axs[1, 0].set_title('State 31')
    axs[1, 0].legend()
    axs[1, 1].plot(x, np.array(q_u)[:, 3], label='u')
    axs[1, 1].plot(x, np.array(q_r)[:, 3], label='r')
    axs[1, 1].plot(x, np.array(q_b)[:, 3], label='b')
    axs[1, 1].plot(x, np.array(q_l)[:, 3], label='l')
    axs[1, 1].set_title('State 32')
    axs[1, 1].legend()
    if isSoftmax:
        fig.suptitle('Backward: Using Softmax' + ' alpha: ' + str(alpha))
    else:
        fig.suptitle('Backward: \u03B5-greedy' + ' (\u03B5=' + str(epsilon)+')')
    fig.show()

else:
    print("Average number of bad episodes ", sum(num_bad_episodes) / len(num_bad_episodes))
    
    x = [i for i in range(1, 1 + len(q_u))]
    fig, axs = plt.subplots(1, 2, figsize = (10, 5))
    
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
    
    '''for i in x_range:
        height = final_q_u[i, 2]
        print(i, height,
                 '$\mu=$%s \n $\sigma=$%s' % (str(round(final_q_u[i, 2], 3)), str(round(final_q_u[i, 3], 3))))'''

    if isSoftmax:
        fig.suptitle('Backward: Using Softmax' + ' alpha: ' + str(alpha) + ' step_size: ' + + str(step_size_1) + ', ' + str(step_size_2))
    else:
        fig.suptitle('Backward: \u03B5-greedy' + ' (\u03B5=' + str(epsilon)+')' + ' step_size: ' + str(step_size_1) + ', ' + str(step_size_2))
    
    plt.savefig('stochastic_graphs/' + title + '.png')
    plt.close(fig)
    
