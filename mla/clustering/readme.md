# Clustering algorithms / Unsupervized Learning

In an unsupervized learning setting, we are given a data matrix $X$ of shape $(n,d)$ and the goal is to split the data points into several clusters (the number may be an input of the algorithm or not).

## K-means



## K-medoids

## Gaussian Mixture Model

Multivariate Gaussian : 
$$
p(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \mathcal{N}\left(\boldsymbol{\mu}, \boldsymbol{\Sigma}\right) = (2 \pi)^{-\frac{D}{2}}|\boldsymbol{\Sigma}|^{-\frac{1}{2}} \exp \left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right)
$$

Expectation step :
$$
r_{nk} = \frac{\pi_{k} \mathcal{N}\left(x_{n} \mid \mu_{k}, \sigma_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(x_{n} \mid \mu_{j}, \sigma_{j}\right)}
$$
$$N_{k} = \sum_{n=1}^{N} r_{n k}$$

Maximization Step :
$$
\boldsymbol{\mu}_{k} =\frac{1}{N_{k}} \sum_{n=1}^{N} r_{n k} \boldsymbol{x}_{n}, \quad
\boldsymbol{\Sigma}_{k} =\frac{1}{N_{k}} \sum_{n=1}^{N} r_{n k}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top}, \quad
\pi_{k} =\frac{N_{k}}{N}
$$


Negative Log Likelihood :

$$
-\sum_{n=1}^{N}\left[\log \left(\frac{1}{\sqrt{2 \pi \sigma^{2}}}\right)-\frac{\left(x_{n}-\mu\right)^{2}}{2 \sigma^{2}}\right]
$$

## DBSCAN

## OPTICS

## Fuzzy C-means
### Input of the Algo
* Data X of shape n,d
* Number of clusters k 
* p the fuzzyness

### Init 
* Weights matrix w of size n,k initialized randomly with condition that $\sum_{j}^k w_{i,j} = 1$

### Computation of the centroids

Iterate while the centroids are changing :
* update the centroids : 
$C_j = \dfrac{\sum_{i}^n w_{i,j}^p X_i}{\sum_{i}^n {i,j}^p}$
* update the weights :
$w_ij = \dfrac{ \dfrac{1}{dist(X_i,C_j)}^{\dfrac{1}{p-1}}}{ \sum_l^k \dfrac{1}{dist(X_i,C_l)}^{\dfrac{1}{p-1}}}$

### Prediction
A data point is assigned a cluster given the probability given by the weight matrix
