import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# simple
MC = pd.read_csv("../results/simple/mc/V_values_0.4.csv")
fwd = pd.read_csv("../results/simple/fwd/V_values_0.4.csv")
bwd = pd.read_csv("../results/simple/BPI/V_values_0.4.csv")

color_1 = "#838B8B"
color_2 = "#9A32CD"
x = np.arange(80000)
fig, axs = plt.subplots(1, 2)
axs[0].plot(x, MC.iloc[:, 1], label='MC (\u03B5 = .07,\u03B1' + r'$_{Q}$' +'=.4,' + r"$\bar{T} = 100$")
axs[0].fill_between(x, MC.iloc[:, 1] - MC.iloc[:, 2], MC.iloc[:, 1] + MC.iloc[:, 2], alpha=0.2)
axs[0].plot(x, fwd.iloc[:,1], color=color_1, label='Soph-EU (\u03B5=.07,\u03B1' + r'$_{Q}$' + '=.4,'+ r"$\bar{T}=100$")
axs[0].fill_between(x, fwd.iloc[:, 1] - fwd.iloc[:, 2], fwd.iloc[:, 1] + fwd.iloc[:, 2], color=color_1, alpha=0.2)
axs[0].plot(x, bwd.iloc[:,1], color=color_2, label='BPI (\u03B5=.07,\u03B1' + r'$_{Q}$' + '=.4,'+ r"$\bar{T}=100$")
axs[0].fill_between(x, bwd.iloc[:, 1] - bwd.iloc[:, 2], bwd.iloc[:, 1] + bwd.iloc[:, 2], color=color_2, alpha=0.2)
axs[0].legend(prop={"size":9})
axs[0].set_title("State 21")

axs[1].plot(x, MC.iloc[:, 3], label='MC (\u03B5 = .07,\u03B1' + r'$_{Q}$' + '=.4,' + r"$\bar{T} = 100$")
axs[1].fill_between(x, MC.iloc[:, 3] - MC.iloc[:, 4], MC.iloc[:, 3] + MC.iloc[:, 4], alpha=0.2)
axs[1].plot(x, fwd.iloc[:,3], color=color_1, label='Soph-EU (\u03B5= .07,' + '\u03B1' + r'$_{Q}$' + '=.4,'+ r"$\bar{T}=100$")
axs[1].fill_between(x, fwd.iloc[:, 3] - fwd.iloc[:, 4], fwd.iloc[:, 3] + fwd.iloc[:, 4], color=color_1, alpha=0.2)
axs[1].plot(x, bwd.iloc[:,3], color=color_2, label='BPI (\u03B5=.07,' + '\u03B1' + r'$_{Q}$' + '=.4,'+ r"$\bar{T}=100$")
axs[1].fill_between(x, bwd.iloc[:, 3] - bwd.iloc[:, 4], bwd.iloc[:, 3] + bwd.iloc[:, 4], color=color_2, alpha=0.2)
axs[1].legend(prop={"size":9})
axs[1].set_title("State 9")
fig.suptitle("Deterministic")
plt.show()

# stochastic
MC_stoc = pd.read_csv("../results/stoc/mc/V_values_0.4.csv")
fwd_stoc = pd.read_csv("../results/stoc/fwd/V_values_0.4.csv")
bwd_stoc = pd.read_csv("../results/stoc/BPI/V_values_0.4.csv")
fig, axs = plt.subplots(1, 2)
axs[0].plot(x, MC_stoc.iloc[:, 1], label='MC (\u03B5 = .07, \u03B1'+ r'$_{Q}$' + '=.4,' + r"$\bar{T} = 100$")
axs[0].fill_between(x, MC_stoc.iloc[:, 1] - MC_stoc.iloc[:, 2], MC_stoc.iloc[:, 1] + MC_stoc.iloc[:, 2], alpha=0.2)
axs[0].plot(x, fwd_stoc.iloc[:,1], color=color_1, label='Soph-EU (\u03B5=.07,' + '\u03B1' + r'$_{Q}$' + '=.4,'+ r"$\bar{T}=100$")
axs[0].fill_between(x, fwd_stoc.iloc[:, 1] - fwd_stoc.iloc[:, 2], fwd_stoc.iloc[:, 1] + fwd_stoc.iloc[:, 2], color=color_1, alpha=0.2)
axs[0].plot(x, bwd_stoc.iloc[:,1], color=color_2, label='BPI (\u03B5=.07,' + '\u03B1' + r'$_{Q}$' + '=.4,'+ r"$\bar{T}=100$")
axs[0].fill_between(x, bwd_stoc.iloc[:, 1] - bwd_stoc.iloc[:, 2], bwd_stoc.iloc[:, 1] + bwd_stoc.iloc[:, 2], color=color_2, alpha=0.2)
axs[0].legend(prop={"size":9})
axs[0].set_title("State 21")

axs[1].plot(x, MC_stoc.iloc[:, 3], label='MC (\u03B5 = .07,\u03B1'+ r'$_{Q}$'+'=.4,' + r"$\bar{T} = 100$")
axs[1].fill_between(x, MC_stoc.iloc[:, 3] - MC_stoc.iloc[:, 4], MC_stoc.iloc[:, 3] + MC_stoc.iloc[:, 4], alpha=0.2)
axs[1].plot(x, fwd_stoc.iloc[:,3], color=color_1, label='Soph-EU (\u03B5=.07,' + '\u03B1'+ r'$_{Q}$' +'=.4,'+ r"$\bar{T}=100$")
axs[1].fill_between(x, fwd_stoc.iloc[:, 3] - fwd_stoc.iloc[:, 4], fwd_stoc.iloc[:, 3] + fwd_stoc.iloc[:, 4], color=color_1, alpha=0.2)
axs[1].plot(x, bwd_stoc.iloc[:,3], color=color_2, label='BPI (\u03B5=.07,\u03B1' + r'$_{Q}$' + '=.4,'+ r"$\bar{T}=100$")
axs[1].fill_between(x, bwd_stoc.iloc[:, 3] - bwd_stoc.iloc[:, 4], bwd_stoc.iloc[:, 3] + bwd_stoc.iloc[:, 4], color=color_2, alpha=0.2)
axs[1].legend(prop={"size":9})
axs[1].set_title("State 9")
fig.suptitle("Stochastic-1")
plt.show()