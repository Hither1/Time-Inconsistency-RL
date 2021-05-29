import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("")

from collections import defaultdict
from envs.DoughVeg_gridworld import GridworldEnv

discount_factor = 0.01
discounting = 'hyper'  # 'hyper', 'exp'
init_policy = 'random'  # 'random' 'stable'

epsilon = .3
num_episodes = 50000  # 0000

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
    global Q_correction_21, Q_21_u, Q_21_r, Q_21_b, Q_21_l, Q_9_u, Q_9_r, Q_9_b, Q_9_l

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # policy is one-to-one to Q: init at UP for each s
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    Returns = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

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

        if is_wall(state):
            # print('init on wall, skip')
            continue

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
        #sa_in_episode = set([(x[0], x[1]) for x in episode])

        sa_set = set({})
        for t in range(len(episode)):
            state, action, reward = episode[t]
            delay = len(episode) - t
            G = discount(delay) * episode[t][2]

            sa_pair = (state, action)
            if sa_pair not in sa_set:
                sa_set.add(sa_pair)
                Q[state][action] = np.mean(Returns[state][action])


        #for state, action in sa_in_episode:



            # Find the first occurence of the (s, a) pair (FROM THE FRONT, correct.)
            # This update is questionable for INCONSISTENT problem.
            first_occurence_idx = next(i for i, x in enumerate(episode)
                                       if x[0] == state and x[1] == action)

            # G = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])

            #sum([x[2] * (discount(i)) for i, x in enumerate(episode[first_occurence_idx:])])

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
                print('curr_policy_9:', curr_policy)

                print('G:', G)

            # if state == 21 and action == 1: # check for when the change to SPE is reflected at 21
            if state == 21 and Q[21][1] > Q[21][0]:

                if critical_episode_21 == 0:  # update the episode in which we reach state 21
                    critical_episode_21 = i_episode - 1

                # count_21 += 1

                curr_policy = np.argmax(policy(state))
                curr_Q = Q[state]
                print('curr_policy:', curr_policy)
                print('curr_Q[21]:', curr_Q)

                print('G:', G)  # prolly need to update from back?

                print('--------------')

        # Track Q[21] for all actions and plot
        Q_21_u += [Q[21][0]]
        Q_21_r += [Q[21][1]]
        Q_21_b += [Q[21][2]]
        Q_21_l += [Q[21][3]]

        # Track Q[9] for all actions and plot
        Q_9_u += [Q[9][0]]
        Q_9_r += [Q[9][1]]
        Q_9_b += [Q[9][2]]
        Q_9_l += [Q[9][3]]

    return Q, policy


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
#
Q_9_u = []
Q_9_r = []
Q_9_b = []
Q_9_l = []
np.random.seed(0)
Q, policy = mc_control_epsilon_greedy(env, num_episodes, discount_factor, epsilon)

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
plt.title("On-policy MC Control: (u - r)")
plt.show()

plt.figure()
plt.plot(x, Q_21_u, label='u')
plt.plot(x, Q_21_r, label='r')
plt.plot(x, Q_21_b, label='b')
plt.plot(x, Q_21_l, label='l')
plt.legend()
plt.title("On-policy MC Control: Q_21")
plt.show()
sum = Q_21_u[len(Q_21_u) - 1] + Q_21_r[len(Q_21_r) - 1] + Q_21_b[len(Q_21_b) - 1] + Q_21_l[len(Q_21_l) - 1]
print("u " + str(Q_21_u[len(Q_21_u) - 1]) + " prob " + str(Q_21_u[len(Q_21_u) - 1] / sum))
print("r " + str(Q_21_r[len(Q_21_r) - 1]) + " prob " + str(Q_21_r[len(Q_21_r) - 1] / sum))
print("b " + str(Q_21_b[len(Q_21_b) - 1]) + " prob " + str(Q_21_b[len(Q_21_b) - 1] / sum))
print("l " + str(Q_21_l[len(Q_21_l) - 1]) + " prob " + str(Q_21_l[len(Q_21_l) - 1] / sum))

sum = Q_9_u[len(Q_9_u) - 1] + Q_9_r[len(Q_9_r) - 1] + Q_9_b[len(Q_9_b) - 1] + Q_9_l[len(Q_9_l) - 1]
print("u " + str(Q_9_u[len(Q_9_u) - 1]) + " prob " + str(Q_9_u[len(Q_9_u) - 1] / sum))
print("r " + str(Q_9_r[len(Q_9_r) - 1]) + " prob " + str(Q_9_r[len(Q_9_r) - 1] / sum))
print("b " + str(Q_9_b[len(Q_9_b) - 1]) + " prob " + str(Q_9_b[len(Q_9_b) - 1] / sum))
print("l " + str(Q_9_l[len(Q_9_l) - 1]) + " prob " + str(Q_9_l[len(Q_9_l) - 1] / sum))

'''Computing V*(s)'''
V_estimate = defaultdict(float)

for state, actions in Q.items():
    action_value = np.max(actions)
    V_estimate[state] = action_value

# Check any scaling issue
avg_V_est = np.average([V_estimate[s] for s in V_estimate.keys()])

'''Computing real \pi*-driven-trajectory values R(\tau)'''
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

    # Total_Reward = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode)])
    Total_Reward = sum([x[2] * (discount(i)) for i, x in enumerate(episode)])

    if s0 == 21:
        print(episode)
        print(Total_Reward)

    V_realized[s0] = Total_Reward
    V_error[s0] = Total_Reward - V_estimate[s0]

# Check any scaling issue
avg_V_real = np.average([V_realized[s] for s in V_realized.keys()])

''' Results'''
print('DISCOUNTING:', discounting)
print('sq_error of value_estimate:', V_error)
print('sum_of_sq_error:', sum([V_error[s] ** 2 for s in V_error.keys()]))
print('averaged values:', avg_V_real)

'''Plot error each episode - see improvement in errors'''

# Hypothesis: THE LONGER THE TRAJECTORY
# -> THE LARGER THE POSSIBILITY OF INCLUDING UNSTABLE CONTINUATION-TRAJECTORY
# -> THE WORST THE ERROR DEVIATION R(\tau(s)) - V(s)



























