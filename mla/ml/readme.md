# Machine Learning Algorithms / Supervized Learning

In supervized learning we are given a data matrix $X$ and target vector $y$ and the goal is to create a function that maps datapoints from $X_i$ to its corresponding target $y_i$.

We distinguish two cases :
* Classification : In this setting the targets are classes draw from a discrete set. The metric used to measure the accuracy of the model can then be simply $\frac{1}{N} \sum_{i=1}^N \mathbb{1}_{\hat y_i = y_i}$.
* Regression : Here the targets are drawn from a continuous set (in opposition to a discrete set). The metric can be thought as a distance between the true target and the predicted one, for example the Mean Squared Loss (MSE) : $\frac{1}{N} \sum_{i=1}^N (\hat y_i - y_i)^2$

## K-Nearest Neighbor (KNN)

With this simple algorithm we only need the number of neighbors $k$ to consider

### Classification
A point is assigned a class by considering the k neareast neighbors (in the training set) and taking the most represented class from those. If there is an equality between two classes we take only the k-1 neighbors.

### Regression
The assigned value of a point is the mean of the k neareast neighbors (in the training set).

## Least Square linear Regression/Classification

The Least Squares aim to minimize the mean squared error : 
$$\hat \beta = \argmin_{\beta} \| y - X\beta \|^2_2$$
This is one of the only algorithm that has a closed form solution : 
$$\hat \beta = (X^T X)^{-1}X^Ty$$

### Polynomial Case
The polynomial case simply add the power of the precised power of each feature to the input data $X$ as follows with a degree equal to 2 :

$$\begin{bmatrix}
x_{1,1} &  ... & x_{1,d}\\
\vdots &  & \vdots \\
x_{n,1} & ... & x_{n,d}
\end{bmatrix}
\rightarrow
\begin{bmatrix}
x_{1,1} &  ... & x_{1,d} & x_{1,1}^2 & ... & x_{1,d}^2\\
\vdots &  & \vdots & \vdots & & \vdots\\
x_{n,1} & ... & x_{n,d} & x_{n,1}^2 & ... &  x_{n,d}^2
\end{bmatrix}
$$

### Ridge / $l_2$ regularized

In this case we add an $l_2$ regularization term  : 
$$\hat \beta = \argmin_{\beta} \| y - X\beta \|^2_2 + \lambda \|\beta\|_2^2$$
This again has a closed form : 
$$ \hat \beta = (X^T X+\lambda \mathbb{I})^{-1}X^Ty$$

$\lambda$ is simply a variable to scale the strengh of the regularization. 

### LASSO / $l_1$ regularized 
In this case we add an $l_1$ regularization term  : 
$$\hat \beta = \argmin_{\beta} \| y - X\beta \|^2_2 + \lambda \|\beta\|_1$$

This has unfortunetly no closed form, two well know solutions exists to estimate the parameter vector $\hat \beta$ :
* Least Angle Regression
* Forward stepwise regressions

#### Least Angle Regression
*(upcoming)* 

## CART Decision Tree
Binary trees can be used as a machine learning model as both regressor and classifier. We will focus on CART decision trees (which have the benefit of being alble to process also categorial data) but other methods exists (ID3, C4.5 ...).

Each node of the tree perform a split on the data using a single variable using a threshold $T$ ($X_d < T$ if variable is numeric or $X_d == T$ if it is categorical). At the bottom of the tree are Leaves which simply output an answer no matter the data given.

### Tree Creation

The tree is created given a dataset $D$, by recursively doing the following step :
* Finding the best split to divide to divide the data based on some metric (gini index for classification and mean squared error for regression), by testing all values of the dataset as a threshold.
* Create two new nodes (left and right) using the splitted data according the best split defined before. 
* A Leaf is created if when creating a node either the maximum depth of the tree is attained or the minimum number of samples is bigger than the number of left datapoints. 
  
### Gini Index

When creating a CART classification decision tree, the gini index is used. The gini index is a cost function to evaluate the split defined as the following : 

$$
G = 1 - \sum_{i \in N} p_{i}^2
$$
with $p_i$ denoting the probability of an element being classified for a distinct class.

## Naive Bayes

### Bernouilli

### Multinomial

### Gaussian

## Logistic Regression / Perceptron

## Discriminant Analysis
### Linear Discriminant analysis

### Quadratic Discriminant analysis

## Gaussian Process Regressor

The GPR uses kernels, the default and most used is the Squared Exponential Kernel (also called RBF), defined as follows : 
$$
\kappa_{y}\left(x_{p}, x_{q}\right)=\sigma_{f}^{2} \exp \left(-\frac{1}{2 \ell^{2}}\left(x_{p}-x_{q}\right)^{2}\right)+\sigma_{n}^{2} \delta_{p q}
$$

We consider we have noise observation scaled by $\sigma_{n}$  :
$$
\mathbf{K}_{y}  \triangleq  \mathbf{K}+\sigma_{n}^{2} \mathbf{I}_{N}
$$
And so the posterior predictive density is :
$$
\begin{aligned}
p\left(\mathbf{f}_{*} \mid \mathbf{X}_{*}, \mathbf{X}, \mathbf{y}\right) &=\mathcal{N}\left(\mathbf{f}_{*} \mid \boldsymbol{\mu}_{*}, \boldsymbol{\Sigma}_{*}\right) \\
\boldsymbol{\mu}_{*} &=\mathbf{K}_{*}^{T} \mathbf{K}_{y}^{-1} \mathbf{y} \\
\boldsymbol{\Sigma}_{*} &=\mathbf{K}_{* *}-\mathbf{K}_{*}^{T} \mathbf{K}_{y}^{-1} \mathbf{K}_{*}
\end{aligned}
$$
with $\mathbf{X}$ the training data, $\mathbf{y}$ the corresponding targets, $\mathbf{X}_*$ the data from which we want to infer $\mathbf{f}_{*}$.