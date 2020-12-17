# Multi-armed bandits

In a Multi-armed Bandits problem, we have a bandit with multiple arms. At each iteration, we choose an arm to activate and we receive a reward. Our goal is then to maximize the global reward (sum of all rewards).    
We can also try to maximize our confidence in which arm is the best (has maximal mean reward) in some cases (you can think for example as a clinic trial in which we prefer to be more confident about the treatment than to maximize the number of cured patients **in** the trial).

Each algo will be run at most $T$ times with a $k$-armed bandit. And the rewards are summarized in the matrix $X_{A_t,t}$ with the arm A at time t.

## Uniform exploration 

A first naive algorithm can be simply implemented as follows : 
* Draw each arm $T/k$ times.

## Follow the Leader (FTL)

This second naive algorithm draws the current best empirical arm : 
* Draw each arm once
* Choose the next arm using : $\hat a = \argmax_a \frac{1}{N_a(t)} \sum_{s=1}^t X_{a,s} \mathbb{1}_{A_s = a}$ 

## Explore then Commit (ETC)

In this simple algorithm we uniformly explore each arm a predetermined number of times and then commit (always choose) the arm with best empirical mean.

It can be implemented as follows :   

Given $m \in \{0,1,...,T/k\}$ 
* Draw each arm m times
* Compute the empirical best arm $\hat a = \argmax_a \hat \mu_a (Km)$
* Draw arm $\hat a$ until round $T$.

## Upper Confidence Bound (UCB)

The UCB-1 algorithm computes for each arm build a confidence interval on the mean and then act as if the best possible model were the true model.
* First draw each arm once.
* Then select the arm using $A_{t+1} = \argmax_a \text{UCB}_a(t)$ with 
$$\text{UCB}_a(t) = \hat \mu_a(t) + \sqrt{\frac{\alpha \log(t)}{N_a(t)}}$$
the first term is the exploitation term (incorporating the empirical means) and the second one the exploration bonus.

## Kullback-leiber UCB (kl-UCB)

## Thompson Sampling

## Linear UCB

LinUCB takes as an input a threshold function $\beta(t,\delta)$ and select the arm using: 
$$A_{t+1}^{\text{LinUCB}} = \underset{a \in \{1,\dots,K\}}{\text{argmax}} \left[x_a^\top \hat{\theta}_t^{\lambda} + \beta(t,\delta) ||x_a||_{\left(B_t^{\lambda}\right)^{-1}}\right]$$

## Linear Thompson Sampling
Thompson Sampling, takes as an input a posterior inflation parameter $v$  : 
$$A_{t+1}^{\text{LinTS}} = \underset{a \in \{1,\dots,K\}}{\text{argmax}} \ x_a^\top \tilde{\theta}_t$$
where $\tilde{\theta}_t$ is a sample from $\mathcal{N}\left(\hat{\theta}_t^{\lambda}, v^2 \left(B_t^{\lambda}\right)^{-1}\right)$.