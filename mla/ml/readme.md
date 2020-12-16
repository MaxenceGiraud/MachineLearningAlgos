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
The polynomial case simply add the power of the precised power of each feature to the input data $X$.

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

*(upcoming)* 

## CART Decision Tree

## Naive Bayes

### Bernouilli

### Multinomial

### Gaussian

## Logistic Regression / Perceptron

## Discriminant Analysis
### Linear Discriminant analysis

### Quadratic Discriminant analysis

