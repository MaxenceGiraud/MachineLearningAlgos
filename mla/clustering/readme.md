# Clustering algorithms / Unsupervized Learning

## K-means

## K-medoids

## DBSCAN


## Fuzzy C-means
### Input of the Algo
* Data X of shape n,d
* Number of clusters k 
* p the fuzzyness

### Init 
* Weights matrix w of size n,k initialized randomly with condition that 
$$ \sum_{j}^k w_{i,j} = 1 $$

### Computation of the centroids

Iterate while the centroids are changing :
* update the centroids : 
$$ C_j = \dfrac{\sum_{i}^n w_{i,j}^p X_i}{\sum_{i}^n {i,j}^p} $$
* update the weights :
$$ w_ij = \dfrac{ \dfrac{1}{dist(X_i,C_j)}^{\dfrac{1}{p-1}}}{ \sum_l^k \dfrac{1}{dist(X_i,C_l)}^{\dfrac{1}{p-1}}} $$

### Prediction
A data point is assigned a cluster given the probability given by the weight matrix

## Gaussian Mixture Model