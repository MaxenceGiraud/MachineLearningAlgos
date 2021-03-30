# Clustering algorithms / Unsupervized Learning

In an unsupervized learning setting, we are given a data matrix $X$ of shape $(n,d)$ and the goal is to split the data points into several clusters (the number may be an input of the algorithm or not).

## K-means

The centroids are initialized either at random or using a smarter technique (e.g. see k-means++ in the next section). 
Then while the centroids are updated, the algorithm iterates : 
* Commpute the distance between all the points to all the centroids.
* Assign the points a cluster given the smallest distance to the centroids.
* Update the centroids by averaging the points of the assigned cluster.

This algorithm is not computationally efficient as it compute $N^2$ at each step, and so is $\mathcal{O}({MN^2})$ with M steps.

## Kmeans++ init
Kmeans++ init is a smart initialization method for unsupervized learning when the number of clusters is known / guessed. 

It finds the centroids as follows : 
* 1. Choose one center uniformly at random among the data points.
* 2. For each data point x not chosen yet, compute D(x), the distance between x and the nearest center that has already been chosen.
* 3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
* 4. Repeat Steps 2 and 3 until k centers have been chosen.

## K-medoids

K-medoids is very similar to K-means but the algorithm only uses points of the dataset as centroids. And so at each step choose the centroid as the closest point in the dataset of the mean of the assigned points in the cluster (instead of simply taking this mean as in k-means).


## Gaussian Mixture Model

In GMMs the centroids are represented as Gaussian distributions with mean $\mu_k$ and covariance matrix $\Sigma_k$. We recall the formula of the Multivariate Gaussian : 
$$
p(\boldsymbol{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \mathcal{N}\left(\boldsymbol{\mu}, \boldsymbol{\Sigma}\right) = (2 \pi)^{-\frac{D}{2}}|\boldsymbol{\Sigma}|^{-\frac{1}{2}} \exp \left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{\top} \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right)
$$

The algorithm then uses the EM algorithm to find the best fitting distribution parameters for the centroids. The EM algorithm simply iterates the following two steps : 
* The Expectation step :
$$
r_{nk} = \frac{\pi_{k} \mathcal{N}\left(x_{n} \mid \mu_{k}, \sigma_{k}\right)}{\sum_{j=1}^{K} \pi_{j} \mathcal{N}\left(x_{n} \mid \mu_{j}, \sigma_{j}\right)}
$$
$$N_{k} = \sum_{n=1}^{N} r_{n k}$$

* The Maximization Step :
$$
\boldsymbol{\mu}_{k} =\frac{1}{N_{k}} \sum_{n=1}^{N} r_{n k} \boldsymbol{x}_{n}, \quad
\boldsymbol{\Sigma}_{k} =\frac{1}{N_{k}} \sum_{n=1}^{N} r_{n k}\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)\left(\boldsymbol{x}_{n}-\boldsymbol{\mu}_{k}\right)^{\top}, \quad
\pi_{k} =\frac{N_{k}}{N}
$$

The algorithm stops when either the maximum number of iteration or the Negative Log Likelihood stops reducing, the nll is in this context defined as follows : 
$$
-\sum_{n=1}^{N}\left[\log \left(\frac{1}{\sqrt{2 \pi \sigma^{2}}}\right)-\frac{\left(x_{n}-\mu\right)^{2}}{2 \sigma^{2}}\right]
$$

## DBSCAN

## OPTICS

## Fuzzy C-means

The fuzzy C-means takes 2 variables as input :
* The number of clusters k 
* The fuzzyness p

The weight matrix w of size n,k initialized randomly with condition that $\sum_{j}^k w_{i,j} = 1$

And then the algorithm iterates the following until the centroids are not changing anymore:
* update the centroids : 
$C_j = \dfrac{\sum_{i}^n w_{i,j}^p X_i}{\sum_{i}^n {i,j}^p}$
* update the weights :
$w_ij = \dfrac{ \dfrac{1}{dist(X_i,C_j)}^{\dfrac{1}{p-1}}}{ \sum_l^k \dfrac{1}{dist(X_i,C_l)}^{\dfrac{1}{p-1}}}$

A data point is then assigned a cluster given the probability given by the weight matrix.