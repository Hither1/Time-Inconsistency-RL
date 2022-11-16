# Reinventing Policy Iteration under Time Inconsistency

This repository contains experimental code supporting the results presented in the paper: 

*Lesmana, Nixie S., Huangyuan Su, and Chi Seng Pun. "Reinventing Policy Iteration under Time Inconsistency." (2022).* [\[OpenReview\]](https://openreview.net/forum?id=bN2vWLTh0P)

> **Abstract:** Policy iteration (PI) is a fundamental policy search algorithm in standard reinforcement learning (RL) setting, which can be shown to converge to an optimal policy by policy improvement theorems. However, the standard PI relies on Bellman’s Principle of Optimality, which might be violated by some specifications of objectives (also known as time-inconsistent (TIC) objectives), such as non-exponentially discounted reward functions. The use of standard PI under TIC objectives has thus been marked with questions regarding the convergence of its policy improvement scheme and the optimality of its termination policy, often leading to its avoidance. In this paper, we consider an infinite-horizon TIC RL setting and formally present an alternative type of optimality drawn from game theory, i.e., subgame perfect equilibrium (SPE), that attempts to resolve the aforementioned questions. We first analyze standard PI under the SPE type of optimality, revealing its merits and insufficiencies. Drawing on these observations, we propose backward Q-learning (bwdQ), a new algorithm in the approximate PI family that targets SPE policy under non-exponentially discounted reward functions. Finally, with two TIC gridworld environments, we demonstrate the implications of our theoretical findings on the behavior of bwdQ and other approximate PI variants.

The algorithms supported are:
- Backward Q-Learning (bwdQ), the original algorithm proposed in our paper.
- Hyperbolic-discounting On-Policy Monte Carlo Control, reimplemented from https://github.com/dennybritz/reinforcement-learning/tree/master/MC.
- Sophisticated Expected-Utility Agent, reimplemented from [\[Evans, 2016\]](https://arxiv.org/abs/1512.05832).
- Backward + Tabular Hyper-Rainbow Policy Evaluation, adapted from [\[Fedus, 2019\]](https://arxiv.org/abs/1902.06865).

The benchmark environments supported are:
- Deterministic Gridworld
- Stochastic Gridworld

For more details, please refer to our paper's Appendix C-D.
