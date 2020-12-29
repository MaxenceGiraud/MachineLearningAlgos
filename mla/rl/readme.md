# Reinforcement Learning

The following explanations are greatly inspired by the notebooks made by [Omar Darwiche Domingues](https://omardrwch.github.io/) as part of the course [Sequential Decision Making](http://chercheurs.lille.inria.fr/ekaufman/SDM.html). The notebooks are available [here](https://github.com/rlberry-py/tutorials).

## Table of Contents 

- [Reinforcement Learning](#reinforcement-learning)
  - [Table of Contents](#table-of-contents)
  - [Introduction - Markov Decision Processes and Value Functions](#introduction---markov-decision-processes-and-value-functions)
  - [Value iteration](#value-iteration)
  - [Q-Learning](#q-learning)
  - [Deep Q-Learning (DQN)](#deep-q-learning-dqn)
  - [Advantage Actor Critic (A2C)](#advantage-actor-critic-a2c)

## Introduction - Markov Decision Processes and Value Functions

In reinforcement learning, an agent interacts with an environment by taking actions and observing rewards. Its goal is to learn a *policy*, that is, a mapping from states to actions, that maximizes the amount of reward it gathers.

The enviroment is modeled as a __Markov decision process (MDP)__, defined by a set of states $\mathcal{S}$, a set of actions $\mathcal{A}$, a reward function $r(s, a)$ and transition probabilities $P(s'|s,a)$. When an agent takes action $a$ in state $s$, it receives a random reward with mean $r(s,a)$ and makes a transion to a state $s'$ distributed according to $P(s'|s,a)$.

A __policy__ $\pi$ is such that $\pi(a|s)$ gives the probability of choosing an action $a$ in state $s$. __If the policy is deterministic__, we denote by $\pi(s)$ the action that it chooses in state $s$. We are interested in finding a policy that maximizes the value function $V^\pi$, defined as 

$$
V^\pi(s) = \sum_{a\in \mathcal{A}} \pi(a|s) Q^\pi(s, a), 
\quad \text{where} \quad 
Q^\pi(s, a) = \mathbf{E}\left[ \sum_{t=0}^\infty \gamma^t r(S_t, A_t)  \Big| S_0 = s, A_0 = a\right].
$$
and represents the mean of the sum of discounted rewards gathered by the policy $\pi$ in the MDP, where $\gamma \in [0, 1[$ is a discount factor ensuring the convergence of the sum. 

The __action-value function__ $Q^\pi$ is the __fixed point of the Bellman operator $T^\pi$__:

$$ 
Q^\pi(s, a) = T^\pi Q^\pi(s, a)
$$
where, for any function $f: \mathcal{S}\times\mathcal{A} \to \mathbb{R}$
$$
T^\pi f(s, a) =  r(s, a) + \gamma \sum_{s'} P(s'|s,a) \left(\sum_{a'}\pi(a'|s')f(s',a')\right) 
$$


The __optimal value function__, defined as $V^*(s) = \max_\pi V^\pi(s)$ can be shown to satisfy $V^*(s) = \max_a Q^*(s, a)$, where $Q^*$ is the __fixed point of the optimal Bellman operator $T^*$__: 

$$ 
Q^*(s, a) = T^* Q^*(s, a)
$$
where, for any function $f: \mathcal{S}\times\mathcal{A} \to \mathbb{R}$
$$
T^* f(s, a) =  r(s, a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} f(s', a')
$$
and there exists an __optimal policy__ which is deterministic, given by $\pi^*(s) \in \arg\max_a Q^*(s, a)$.


## Value iteration

If both the reward function $r$ and the transition probablities $P$ are known, we can compute $Q^*$ using value iteration, which proceeds as follows:

1. Start with arbitrary $Q_0$, set $t=0$.
2. Compute $Q_{t+1}(s, a) = T^*Q_t(s,a)$ for every $(s, a)$.
3. If $\max_{s,a} | Q_{t+1}(s, a) -  Q_t(s,a)| \leq \varepsilon$, return $Q_{t}$. Otherwise, set $t \gets t+1$ and go back to 2. 

The convergence is guaranteed by the contraction property of the Bellman operator, and $Q_{t+1}$ can be shown to be a good approximation of $Q^*$ for small epsilon. 


## Q-Learning

Using Q-Learning we don't need to know the transition probabilities, we can approximate $Q^*$ using *samples* from the environment with the Q-Learning algorithm. This method only works with finite state and action space.

Q-Learning with __$\varepsilon$-greedy exploration__ proceeds as follows:

1. Start with arbitrary $Q_0$, get starting state $s_0$, set $t=0$.
2. Choosing action $a_t$: 
  * With probability $\varepsilon$ choose $a_t$ randomly (uniform distribution)  
  * With probability $1-\varepsilon$, choose $a_t \in \arg\max_a Q_t(s_t, a)$.
3. Take action $a_t$, observe next state $s_{t+1}$ and reward $r_t$.
4. Compute error $\delta_t = r_t + \gamma \max_a Q_t(s_{t+1}, a) - Q_t(s_t, a_t)$.
5. Update 
  * $Q_{t+1}(s, a) = Q_t(s, a) + \alpha_t(s,a) \delta_t$,  __if $s=s_t$ and $a=a_t$__
  * $Q_{t+1}(s, a) = Q_{t}(s, a)$ otherwise.

Here, $\alpha_t(s,a)$ is a learning rate that can depend, for instance, on the number of times the algorithm has visited the state-action pair $(s, a)$. 

## Deep Q-Learning (DQN)

Deep Q-Learning used a neural network to approximate $Q$ functions. On the contrary of Q-Learning, with DQN we can have continuous state and action spaces.

The parameters of the neural network are denoted by $\theta$. 
*   As input, the network takes a state $s$,
*   As output, the network returns $Q(s, a, \theta)$, the value of each action $a$ in state $s$, according to the parameters $\theta$.


The goal of Deep Q-Learning is to learn the parameters $\theta$ so that $Q(s, a, \theta)$ approximates well the optimal $Q$-function $Q^*(s, a)$. 

In addition to the network with parameters $\theta$, the algorithm keeps another network with the same architecture and parameters $\theta^-$, called **target network**.

The algorithm works as follows:

*  At each time $t$, the agent is in state $s_t$ and has observed the transitions $(s_i, a_i, r_i, s_i')_{i=1}^{t-1}$, which are stored in a **replay buffer**.

*  Choose action $a_t = \arg\max_a Q(s_t, a)$ with probability $1-\varepsilon_t$, and $a_t$=random action with probability $\varepsilon_t$. 

* Take action $a_t$, observe reward $r_t$ and next state $s_t'$.

* Add transition $(s_t, a_t, r_t, s_t')$ to the **replay buffer**.

*  Sample a minibatch $\mathcal{B}$ containing $B$ transitions from the replay buffer. Using this minibatch, we define the loss:

$$
L(\theta) = \sum_{(s_i, a_i, r_i, s_i') \in \mathcal{B}}
\left[
Q(s_i, a_i, \theta) -  y_i
\right]^2
$$
where the $y_i$ are the **targets** computed with the **target network** $\theta^-$:

$$
y_i = r_i + \gamma \max_{a'} Q(s_i', a', \theta^-).
$$

* Update the parameters $\theta$ to minimize the loss, e.g., with gradient descent (**keeping $\theta^-$ fixed**): 
$$
\theta \gets \theta + \eta \nabla_\theta L(\theta)
$$
where $\eta$ is the optimization learning rate. 

* Every $N$ transitions ($t\mod N$ = 0), update target parameters: $\theta^- \gets \theta$.

* $t \gets t+1$. Stop if $t = T$, otherwise go to step 2.

## Advantage Actor Critic (A2C)

A2C keeps two neural networks:
*   One network with paramemeters $\theta$ to represent the policy $\pi_\theta$.
*   One network with parameters $\omega$ to represent a value function $V_\omega$, that approximates $V^{\pi_\theta}$


At each iteration, A2C collects $M$ transitions $(s_i, a_i, r_i, s_i')_{i=1}^M$ by following the policy $\pi_\theta$. If a terminal state is reached, we simply go back to the initial state and continue to play $\pi_\theta$ until we gather the $M$ transitions.

Consider the following quantities, defined based on the collected transitions:

$$
\widehat{V}(s_i) = \widehat{Q}(s_i, a_i) = \sum_{t=i}^{\tau_i \wedge M} \gamma^{t-i} r_t + \gamma^{M-i+1} V_\omega(s_M')\mathbb{I}\{\tau_i>M\}
$$

where and $\tau_i = \min\{t\geq i: s_i' \text{ is a terminal state}\}$, and 

$$
\mathbf{A}_\omega(s_i, a_i) = \widehat{Q}(s_i, a_i) -  V_\omega(s_i)  
$$


A2C then takes a gradient step to minimize the policy "loss" (keeping $\omega$ fixed):

$$
L_\pi(\theta) =
-\frac{1}{M} \sum_{i=1}^M \mathbf{A}_\omega(s_i, a_i) \log \pi_\theta(a_i|s_i)
- \frac{\alpha}{M}\sum_{i=1}^M \sum_a  \pi(a|s_i) \log \frac{1}{\pi(a|s_i)}
$$

and a gradient step to minimize the value loss (keeping $\theta$ fixed):

$$
L_v(\omega) = \frac{1}{M} \sum_{i=1}^M \left( \widehat{V}(s_i) - V_\omega(s_i)   \right)^2
$$