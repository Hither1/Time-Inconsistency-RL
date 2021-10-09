import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")
sys.path.append("")

from collections import defaultdict
from envs.DoughVeg_gridworld import GridworldEnv
#from envs.DoughVeg_simple_stochastic import GridworldEnv
step_size = .4
discount_factor = 1
discounting = 'hyper'  # 'hyper', 'exp'
init_policy = 'random'  # 'random' 'stable'

epsilon = .07
num_episodes = 80000  # 0000

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


def mc_control_epsilon_greedy(env, num_episodes, discount_factor, epsilon, step_size):
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
        policy(9)[0] = epsilon / 4
        policy(9)[3] += 1 - epsilon

        # The type of discounting
    discount = auto_discounting()

    for i_episode in range(1, num_episodes + 1):

        # Debugging
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        episode = []
        state = env.reset()  # s0 is randomized uniformly

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

            # G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
            G = sum([x[2] * (discount(i+1)) for i, x in enumerate(episode[first_occurence_idx:])])

            # Calculate average return for this state over all sampled episodes
            returns_sum[sa_pair] += G
            returns_count[sa_pair] += 1.0

            Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

            if state == 9 and action == 0:  # check for possible JUMP at 9 due to noisy policy

                if critical_episode_9 == 0:
                    critical_episode_9 = i_episode - 1

                # count_9 += 1

                curr_policy = np.argmax(policy(state))
                curr_Q = Q[state]
                #print('curr_policy_9:', curr_policy)
                #print('curr_Q[9]:', curr_Q)

                #print('G:', G)

            # if state == 21 and action == 1: # check for when the change to SPE is reflected at 21
            if state == 21 and Q[21][1] > Q[21][0]:

                if critical_episode_21 == 0:  # update the episode in which we reach state 21
                    critical_episode_21 = i_episode - 1

                # count_21 += 1

                curr_policy = np.argmax(policy(state))


        # Track Q[21] for all actions and plot
        q_u.append([Q[x][0] for x in range(24)])
        q_r.append([Q[x][1] for x in range(24)])
        q_b.append([Q[x][2] for x in range(24)])
        q_l.append([Q[x][3] for x in range(24)])
        V.append([max(Q[21])])

    return Q


# Critical to check how sensitive is relevant states (i.e. 13, 17, 21) to sudden deviation at 9
critical_states_update_order = []
critical_index_9 = 0
critical_episode_9 = 0
critical_index_21 = 0
critical_episode_21 = 0

Q_correction_21 = []

np.random.seed(0)
# Q, policy = mc_control_epsilon_greedy(env, num_episodes, discount_factor, epsilon)

V_s, q_u_s, q_r_s, q_b_s, q_l_s = [], [], [], [], []
for _ in range(50):
    V, q_u, q_r, q_b, q_l = [], [], [], [], []
    Q = mc_control_epsilon_greedy(env, num_episodes, discount_factor, epsilon, step_size=step_size)
    V, q_u, q_r, q_b, q_l = np.array(V), np.array(q_u), np.array(q_r), np.array(q_b), np.array(q_l)

    q_u_s.append(q_u)
    q_r_s.append(q_r)
    q_b_s.append(q_b)
    q_l_s.append(q_l)
    V_s.append(V)

V_s, q_u_s, q_r_s, q_b_s, q_l_s = np.array(V_s), np.array(q_u_s), np.array(q_r_s), np.array(q_b_s), np.array(q_l_s)


final_q_u, final_q_r, final_q_b, final_q_l, final_V = [], [], [], [], []
for i in range(num_episodes):
    final_V.append([np.mean(V_s[:, i, 0]), np.std(V_s[:, i, 0])])
    final_q_u.append([np.mean(q_u_s[:, i, x]) for x in range(24)] + [np.std(q_u_s[:, i, x]) for x in range(24)])
    final_q_r.append([np.mean(q_r_s[:, i, x]) for x in range(24)] + [np.std(q_r_s[:, i, x]) for x in range(24)])
    final_q_b.append([np.mean(q_b_s[:, i, x]) for x in range(24)] + [np.std(q_b_s[:, i, x]) for x in range(24)])
    final_q_l.append([np.mean(q_l_s[:, i, x]) for x in range(24)] + [np.std(q_l_s[:, i, x]) for x in range(24)])
final_V, final_q_u, final_q_r, final_q_b, final_q_l = np.array(final_V), np.array(final_q_u), np.array(final_q_r), np.array(final_q_b), np.array(final_q_l)


df = pd.DataFrame(final_V)
df.to_csv("../../../results/simple/MC/V_values_" + str(step_size) + ".csv")

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



'''
print("Final Best Actions:")
nr = env.shape[0]
nc = env.shape[1]
for r_ in range(nr):
    row = []
    for c_ in range(nc):
        row.append(np.argmax([final_q_u[-1][r_ * nc + c_],
                            final_q_r[-1][r_ * nc + c_],
                            final_q_b[-1][r_ * nc + c_],
                            final_q_l[-1][r_ * nc + c_]]
                            ))
    print(row)




'''
# Graphs
x = [i for i in range(1, 1 + len(q_u))]

# second pic
'''plt.plot(x, final_V[:, 0], label='u')
plt.fill_between(x, final_V[:, 0] - final_V[:, 1], final_V[:, 0] + final_V[:, 1], alpha=0.2)
print("(21, up)", np.array(q_u)[:, 0][-1])
print("(21, right)", np.array(q_r)[:, 0][-1])
print("(21, below)", np.array(q_b)[:, 0][-1])
print("(21, left)", np.array(q_l)[:, 0][-1])
plt.title('Q(s=21) Gridworld')
plt.legend()
plt.show()

plt.plot(x, final_V[:, 2], label='u')
plt.fill_between(x, final_V[:, 2] - final_V[:, 3], final_V[:, 2] + final_V[:, 3], alpha=0.2)'''
plt.title('Q(s=9) Gridworld')
plt.legend()
plt.suptitle('MC: \u03B5-greedy' + ' (\u03B5=' + str(epsilon) + ')')
plt.legend()
plt.show()

for r in [19999, 29999, 99999]: #999
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

    x = [i for i in range(1 + r)]
    linewidth = 3
    plt.ylim([0, 4])
    plt.xlim([0, r])
    plt.plot(x, final_q_u[:r+1, 21], label='UP', linewidth=linewidth)
    plt.fill_between(x, final_q_u[:r+1, 21] - final_q_u[:r+1, 21+24], final_q_u[:r+1, 21] + final_q_u[:r+1, 21+24], alpha=0.2)
    print("(21, up)",final_q_u[:r+1, 21][-1])
    plt.plot(x, final_q_r[:r+1, 21], label='RIGHT', linewidth=linewidth)
    plt.fill_between(x, final_q_r[:r+1, 21] - final_q_r[:r+1, 21+24], final_q_r[:r+1, 21] + final_q_r[:r+1, 21+24], alpha=0.2)
    print("(21, right)", final_q_r[:r+1, 21][-1])
    plt.title('Q(s=21) MC '+'(\u03B5 = .07,' + r"$\bar{T} = 100$)" + " Deterministic")
    plt.legend()
    plt.annotate("Overtake at " + str(i_21), (i_21, final_q_r[i_21][21] - 0.3))
    plt.show()

    plt.xlim([0, r])
    plt.ylim([0, 6.1])
    plt.plot(x, final_q_u[:r+1, 9], label='UP', linewidth=linewidth)
    plt.fill_between(x, final_q_u[:r+1, 9] - final_q_u[:r+1, 9+24], final_q_u[:r+1, 9] + final_q_u[:r+1, 9+24], alpha=0.2)

    plt.plot(x, final_q_l[:r+1, 9], label='LEFT', color="green", linewidth=linewidth)
    plt.fill_between(x, final_q_l[:r+1, 9] - final_q_l[:r+1, 9+24], final_q_l[:r+1, 9] + final_q_l[:r+1, 9+24], color="green", alpha=0.2)
    plt.title('Q(s=9) MC '+'(\u03B5 = .07,' + r"$\bar{T} = 100$)" + " Deterministic")
    plt.annotate("Overtake at " + str(i_9), (i_9, final_q_u[i_9][9] - 0.3))
    plt.legend()
    plt.show()




























