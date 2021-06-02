# Experiment Instructions
### This file intends to provide detailed instructions on how to reproduce the experiments in our [paper]().
The main setup of the experiments consists of two parts. The first is algorithms, and the second is the environment. We also provide an explanation on how to collect the result data.

## 1. Algorithms
We have used 3 algorithms in this paper. The Vanilla method, _i.e._ Monte Carlo, and the recursive methods, _i.e._, the Soph-EU-Agent and the E. The pseudo-code of the 3 algorithms are included in the Appendix A of our paper. Each of these algorithms is packed into a function, with the inputs to the algorithms being the inputs to the functions. 

For all the 3 algorithms, we use ε-greedy as the randomization method of soft-greedy policy.

### 1.1 MC
For this method, we modify the standard textbook implementation of on-policy MC method to use hyperbolic discounting. The modification is mainly in the way that `G` is updated. In our modified algorithm, we use the following formula to update the `G` factor: 
```
d = T - t
G = 1/(1+k*d) * R_t
```
where `t` is the current time, `T` is the length of the episode, and `R_t` is the reward at time `t`.


### 1.2 Soph-EU-Agent (Forward)
This is the method that we refer to as 'forward update'. 

For the initialization of this algorithm, we define `Utility` as a simple dictionary whole value only depends on the state. Since the gridworld is simple that 
`ExpectedUtility` as a dictionary with 3-layer key. The 3 layers of keys are state `s`, delay `d` and action `a`, respectively.

Empirically, 35000 episodes is enough for this algorithm to converge on the problem of simple gridworld.

#### 1.3.1 Adding Penalty
In order to implement the penalty, we modify the definition of `Utility` such that `Utility(s, a) = 0` for any states `s` that is not a goal state. We leave the value of the goal states unchanged.


### 1.3 Equilbrium Q-Iteration (Backward)
This is the method that we refer to as 'backward update'. 

The `Q` is initialized as a dictionary with 2-layer key, which are state `s` and action `a`. We initialize `f` as a dictionary with 3-layer key. The 3 layers of keys are current time `t`, state `s` and action `a` respectively.

Empirically, 20000 episodes is enough for this algorithm to converge on the problem of simple gridworld.

#### 1.3.1 Adding Penalty
In order to implement the penalty, we need another adjustment function, which is implemented as a 4-layer key. Now, the 4 layers of keys are `m`: the time at which the reward is received, `n`: the current time, state `s` and action `a`. 



## 2. Environment
In this part, we describe how to implement the environments.

### 2.1 Simple Gridworld
Our devise an gridworld environment as shown in the following graph:
By inheriting from the env of , you can just specify the width and height of the grid, and the states will be automatically indexed as shown in the following graph:

<div>
<img src="figs/envs/gridworld.png" width="200" height="280"/>
</div>

In the code, we implement the gridworld as shown by the following: 

    o  o  S  o
    o  o  X  o
    M  o  X  o
    o  o  X  o
    o  o  X  o
    o  S  o  o
    
 where M: Moon (`s=8`), S: Sun (`s=2`), X: Wall, S: Start/Home (`s=21`).

<div>
<img src="figs/envs/gridworld_with_traj.png" width="300" height="280"/>
</div>

### ~~2.2 Windy Gridworld~~





## 3. Result Collection

### 3.1 Q/Expected Utility Values
We mainly measure the Q or Expected Utility values at states 9 and 21. For both cases, we initialize 4 empty lists for the 4 directions of [UP, RIGHT, DOWN, LEFT] to store the Q/Expected Utility values at the end of each episode. For Q values, we record 

For the experiments of Soph-Agent and Equilibrium Q, we repeat each experiemnt for 10 times and calculate the mean and std.dev (standard deviation) for . Then in the resulting graphs, we represent the mean as the center line, and represent the std.dev by shaded areas around the center line.

### 3.2 Number of Revisits

The number of **Revisits** is a new metric that we defined in Section 4. to serve as a measure of efficiency for the algorithms.

To calculate this metric, we first . This is to be used for storing all the numbers of _Revisits_ for each episode. 
For each epsiode, we use a set `{}` to keep track of the states that have been visited in the current episode. Then, throughout each episode, if we encounter a state that we have visited before; otherwise, we add the .
