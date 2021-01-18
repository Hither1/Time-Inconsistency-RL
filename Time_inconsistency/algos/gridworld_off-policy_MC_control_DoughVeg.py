import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append("C:\\Users\\NLESM\\Dropbox\\GradSchool\\RESEARCH\\RLImplementation\Gridworld")

from collections import defaultdict
from envs.DoughVeg_gridworld import GridworldEnv

discount_factor = 1
discounting = 'hyper' #'hyper', 'exp'
init_policy = 'random' #'random' 'stable'

epsilon = .1
num_episodes = 50000 #0000
    
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

def create_random_policy(nA):
    """
    Creates a random policy function.
    
    Args:
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities
    """
    A = np.ones(nA, dtype=float) / nA
    
    def policy_fn(observation):
        return A
    
    return policy_fn

def create_greedy_policy(Q):
    """
    Creates a greedy policy based on Q values.

    Args:
        Q: A dictionary that maps from state -> action values
        
    Returns:
        A function that takes an observation as input and returns a vector
        of action probabilities.
    """
    
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    
    return policy_fn

def off_policy_mc_control(env, num_episodes, behavior_policy):
    """
    Monte Carlo Control Off-Policy Control using Weighted Importance Sampling.
    Finds an optimal greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        behavior_policy: The behavior to follow while generating episodes.
            A function that given an observation returns a vector of probabilities for each action.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities. This is the optimal greedy policy.
    """
    global critical_episode_9, critical_episode_21
    global Q_correction_21, Q_21_u, Q_21_r, Q_21_b, Q_21_l

    discount = auto_discounting()

    # The final action-value function.
    # A dictionary that maps state -> action values
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # The cumulative denominator of the weighted importance sampling formula
    # (across all episodes)
    C = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # Our greedily policy we want to learn
    target_policy = create_greedy_policy(Q)
        
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
            #print('init on wall, skip')
            continue
        for t in range(100):
            # Sample an action from our policy
            probs = behavior_policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break

            if state == 9 and action == 0:  # check for possible JUMP at 9 due to noisy policy

                if critical_episode_9 == 0:
                    critical_episode_9 = i_episode - 1

                # count_9 += 1

                curr_policy = np.argmax(target_policy(state))
                curr_Q = Q[state]
                print('curr_policy_9:', curr_policy)
                print('curr_Q[9]:', curr_Q)

                print('G:', G)

                print('-----')

                state = next_state # Update state to the next
            # if state == 21 and action == 1: # check for when the change to SPE is reflected at 21
            if state == 21 and Q[21][1] > Q[21][0]:

                if critical_episode_21 == 0:  # update the episode in which we reach state 21
                    critical_episode_21 = i_episode - 1

                    # count_21 += 1

                curr_policy = np.argmax(target_policy(state))
                curr_Q = Q[state]
                print('curr_policy:', curr_policy)
                print('curr_Q[21]:', curr_Q)

                print('G:', G)  # prolly need to update from back?

                #print('trajectory aft 21:', episode[first_occurence_idx:])

                print('--------------')
        
        # Sum of discounted returns
        G = 0.0
        # The importance sampling ratio (the weights of the returns)
        W = 1.0
        # For each step in the episode, backwards
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            # Update the total reward since step t
            print("episode number is " + str(i_episode))
            print("t is " + str(t))
            print("discount is " + str(discount(t)))
            print("reward's " + str(reward))
            G = discount(t) * G + reward
            # Update weighted importance sampling formula denominator
            C[state][action] += W
            # Update the action-value function using the incremental update formula (5.7)
            # This also improves our target policy which holds a reference to Q
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            # If the action taken by the behavior policy is not the action 
            # taken by the target policy the probability will be 0 and we can break
            if action != np.argmax(target_policy(state)):
                break
            W = W * 1./behavior_policy(state)[action]

        # Track Q[21] for all actions and plot
        Q_21_u += [Q[21][0]]
        Q_21_r += [Q[21][1]]
        Q_21_b += [Q[21][2]]
        Q_21_l += [Q[21][3]]
    return Q, target_policy


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


random_policy = create_random_policy(env.action_space.n)
Q, policy = off_policy_mc_control(env, num_episodes=2000, behavior_policy=random_policy)

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

x = [i for i in range(1, 1+len(Q_21_u))]

plt.figure()
plt.plot(x, np.array(Q_21_u) - np.array(Q_21_r))
plt.title("Off-policy MC Control: (u - r)")
plt.show()

plt.figure()
plt.plot(x, Q_21_u, label = 'u')
plt.plot(x, Q_21_r, label = 'r')
plt.plot(x, Q_21_b, label = 'b')
plt.plot(x, Q_21_l, label = 'l')
plt.legend()
plt.title("Off-policy MC Control: Q_21")
sum = Q_21_u[len(Q_21_u)-1] + Q_21_r[len(Q_21_r)-1] + Q_21_b[len(Q_21_b)-1] + Q_21_l[len(Q_21_l)-1]
print(Q_21_u[len(Q_21_u)-1] / sum)
print(Q_21_r[len(Q_21_r)-1] / sum)
print(Q_21_b[len(Q_21_b)-1] / sum)
print(Q_21_l[len(Q_21_l)-1] / sum)
plt.show()

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
        
    #Total_Reward = sum([x[2]*(discount_factor**i) for i,x in enumerate(episode)])
    Total_Reward = sum([x[2]*(discount(i)) for i, x in enumerate(episode)])
    
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
print('sum_of_sq_error:', sum([V_error[s]**2 for s in V_error.keys()]))
print('averaged values:', avg_V_real)

'''Plot error each episode - see improvement in errors'''

# Hypothesis: THE LONGER THE TRAJECTORY 
# -> THE LARGER THE POSSIBILITY OF INCLUDING UNSTABLE CONTINUATION-TRAJECTORY 
# -> THE WORST THE ERROR DEVIATION R(\tau(s)) - V(s)





























