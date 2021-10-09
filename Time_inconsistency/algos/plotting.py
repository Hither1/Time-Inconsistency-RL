import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")

# simple
MC = pd.read_csv("../results/simple/mc/V_values_0.4.csv")
fwd = pd.read_csv("../results/simple/fwd/V_values_0.4.csv")
bwd = pd.read_csv("../results/simple/BPI/V_values_0.4.csv")
bwd_re = pd.read_csv("../results/reversed/BPI/V_values_0.4.csv")
color_1 = "tan"  # for Soph-EU
color_2 = "darkturquoise"  # for BPI
color_3 = "black" #"#003333"  # for MC
num_episode = 10000
x = np.arange(num_episode)
default = [2.111] * num_episode

#plt.set_axisbelow(True) # grid in background
#plt.yaxis.grid(color='gray', linestyle='dashed')
linewidth = 3
plt.xlim([0, num_episode])
plt.ylim([0, 4])
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.plot(x, bwd.iloc[:num_episode,1], linewidth=linewidth, color=color_2, label='Backward Q-learning')# (\u03B5=.07,\u03B1' + r'$_{Q}$' + '=.4,'+ r"$\bar{T}=100$")
plt.fill_between(x, bwd.iloc[:num_episode, 1] - bwd.iloc[:num_episode, 2], bwd.iloc[:num_episode, 1] + bwd.iloc[:num_episode, 2], color=color_2, alpha=0.2)
plt.plot(x, fwd.iloc[:num_episode,1], linewidth=linewidth, color=color_1, label='Soph-EU') # (\u03B5=.07,\u03B1' + r'$_{Q}$' + '=.4,'+ r"$\bar{T}=100$")
plt.fill_between(x, fwd.iloc[:num_episode, 1] - fwd.iloc[:num_episode, 2], fwd.iloc[:num_episode, 1] + fwd.iloc[:num_episode, 2], color=color_1, alpha=0.2)
plt.plot(x, MC.iloc[:num_episode, 1], linewidth=linewidth, label='MC', color=color_3)# (\u03B5 = .07,\u03B1' + r'$_{Q}$' +'=.4,' + r"$\bar{T} = 100$")
plt.fill_between(x, MC.iloc[:num_episode, 1] - MC.iloc[:num_episode, 2], MC.iloc[:num_episode, 1] + MC.iloc[:num_episode, 2], color=color_3, alpha=0.2)
plt.plot(x, bwd_re.iloc[:num_episode, 1], linewidth=linewidth, label='Bwd Q-learning (with std cond.)')# (\u03B5 = .07,\u03B1' + r'$_{Q}$' +'=.4,' + r"$\bar{T} = 100$")
plt.fill_between(x, bwd_re.iloc[:num_episode, 1] - bwd_re.iloc[:num_episode, 2], bwd_re.iloc[:num_episode, 1] + bwd_re.iloc[:num_episode, 2], alpha=0.2)


plt.plot(x, default, '--', linewidth=2, color="red",
         label='TRUE')
plt.legend(prop={"size":9})
plt.title("Deterministic State 21")
plt.legend(prop={"size":9})
plt.show()

# stochastic\
plt.xlim([0, num_episode])
plt.ylim([0, 3])
plt.xlabel('Episode')
plt.ylabel('Reward')
MC_stoc = pd.read_csv("../results/stoc/mc/V_values_0.4.csv")
fwd_stoc = pd.read_csv("../results/stoc/fwd/V_values_0.4.csv")
bwd_stoc = pd.read_csv("../results/stoc/BPI/V_values_0.4.csv")
#plt.set_axisbelow(True)
#plt.yaxis.grid(color='gray', linestyle='dashed')
 #
plt.plot(x, bwd_stoc.iloc[:num_episode, 1], color=color_2, linewidth=linewidth,
         label='Backward Q-learning') # (\u03B5=.07,' + '\u03B1' + r'$_{Q}$' + '=.4,' + r"$\bar{T}=100$")
plt.fill_between(x, bwd_stoc.iloc[:num_episode, 1] - bwd_stoc.iloc[:num_episode, 2], bwd_stoc.iloc[:num_episode, 1] + bwd_stoc.iloc[:num_episode, 2], color=color_2,
                 alpha=0.2)
plt.plot(x, fwd_stoc.iloc[:num_episode, 1], color=color_1, linewidth=linewidth,
         label='Soph-EU') # (\u03B5=.07,' + '\u03B1' + r'$_{Q}$' + '=.4,' + r"$\bar{T}=100$")
plt.fill_between(x, fwd_stoc.iloc[:num_episode, 1] - fwd_stoc.iloc[:num_episode, 2], fwd_stoc.iloc[:num_episode, 1] + fwd_stoc.iloc[:num_episode, 2], color=color_1,
                 alpha=0.2)

plt.plot(x, MC_stoc.iloc[:num_episode, 1], linewidth=linewidth,
         label='MC', color=color_3) # (\u03B5 = .07, \u03B1' + r'$_{Q}$' + '=.4,' + r"$\bar{T} = 100$")
plt.fill_between(x, MC_stoc.iloc[:num_episode, 1] - MC_stoc.iloc[:num_episode, 2], MC_stoc.iloc[:num_episode, 1] + MC_stoc.iloc[:num_episode, 2], alpha=0.3)
plt.plot(x, default, '--', linewidth=2, color="red",
         label='TRUE')
plt.legend(prop={"size": 9})
plt.title("Stochastic State 21")

plt.show()
