# MachineLearningAlgos
Personal reimplementation of some ML algorithm for learning purposes

## Author
[Maxence Giraud](https://github.com/MaxenceGiraud/)

## Requirements 
* [NumPy](https://numpy.org/) 
* [Matplotlib](https://matplotlib.org/) (for plots only)
* [Gym](https://gym.openai.com/) (RL environment)

## How to
Some examples on how to use the algorithms can be found in the file example.py

## Algorithms

### [Machine Leaning](./mla/ml)
- [x] [KNN](./mla/ml/knn.py) (classifier+regressor)
- [x] [Logistic Regression/ Perceptron](./mla/ml/perceptron.py)
- [x] [Least squares](./mla/ml/leastsquares.py) linear regression/classification
- [x] [Least squares](./mla/ml/leastsquares.py) polynomial Regression/classification
- [x] [Ridge](./mla/ml/leastsquares_regularised.py) 
- [ ] [LASSO](./mla/ml/leastsquares_regularised.py) (using Least Angle Regression)
- [ ] [SVM](./mla/ml/svm.py)
- [x] [CART Decision Tree](./mla/ml/decison_tree.py)
- [ ] Gaussian process regression
- [ ] Hidden Markov model
- [ ] Multivariate adaptive regression spline

#### [Bayesian](./mla/ml)
- [x] [Naive Bayes](./mla/ml/naive_bayes.py) (Bernoulli, Gaussian and Multinomial)
- [x] [LDA/QDA](./mla/ml/discriminantanalysis.py)
- [ ] Variational Bayes - Laplace approx (Murphy 8.4.1-2)
- [ ] MCMC (Murphy chap24)
- [ ] Importance sampling (Murphy chap23)
- [ ] SMC
  

### [Ensemble methods](./mla/ensemble/)
- [x] [Random Forest](./mla/ensemble/random_forest.py)
- [ ] Gradient Boosting
- [ ] AdaBoost
- [x] [One vs rest classifier](./mla/ensemble/multiclass.py)
- [x] [One vs one classifier ](./mla/ensemble/multiclass.py)
  

### [Clustering](./mla/clustering/)
- [x] [K-means](./mla/clustering/kmeans.py)
- [x] [K-medoids](./mla/clustering/kmedoids.py)
- [x] [DBSCAN](./mla/clustering/dbscan.py)
- [ ] OPTICS
- [ ] Gaussian Mixture with EM algorithm
- [ ] Variational Bayesian Gaussian Mixture
- [ ] Generative topographic map
- [ ] Vector quantization
- [ ] Fuzzy clustering (ex : Fuzzy C-means)
  

### [Deep Leaning](./mla/dl/)
- [x] [Neural Network Base/sequential](./mla/dl/neuralnetwork.py)

#### [Layers](./mla/dl/layers/)
- [x] [Dense](./mla/dl/layers/dense.py)
- [ ] Convolution (1d,2d,3d)
- [ ] Recurrent
- [ ] LSTM
- [ ] Pooling
- [x] Flatten
- [ ] Dropout
- [ ] Normalization

#### Wrapper
- [ ] Autoencoder
- [ ] VAE
- [ ] GAN 
- [ ] Transformer 
  
#### [Optimization Algorithm](./mla/dl/optimizer/)
- [x] Stochastic gradient descent
- [x] Mini batch batch Gradient descent
- [ ] Adam
- [ ] Adagrad

#### [Loss](./mla/dl/layers/loss.py)
- [x] MSE
- [x] MAE
- [x] Binary Cross Entropy
- [ ] Neg log likelihood  
- [ ] Add possibility of weighted loss
  
#### [Activation functions](./mla/dl/activation/activation.py)
- [x] Linear
- [x] Relu
- [x] Tanh
- [x] Sigmoid
- [x] Leaky Relu
- [x] Softplus
- [ ] Softmax
- [ ] Softmin

### [Multi-armed bandit](./mla/mab/)
Credits to [Emilie Kaufmann](http://chercheurs.lille.inria.fr/ekaufman/index.html) for the bandit environment. 

- [x] UCB
- [x] kl UCB
- [x] ETC 
- [x] FTL
- [x] Thompson sampling
- [ ] Lin UCB
- [ ] Lin Thompsom Sampling

### Reinforcement Learning 
- [ ] Q-learning
  
### Data processing/Analysis
- [ ] PCA
- [ ] FDA
- [ ] ICA
- [ ] Elastic map/net
- [ ] CCA

### Visualization
- [ ] Draw Decision boundary(ies)
- [ ] Confusion matrix ?


## References


[1] M. BISHOP, Christopher. Pattern Recognition and Machine Learning. Springer, 2011.   
[2] TIBSHIRANI, Robert, HASTIE, Trevor y FRIEDMAN JEROME, . The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition. Springer, 2016.   
[3] P. MURPHY, Kevin. Machine Learning: A Probabilistic Perspective. The MIT Press, 2012.   
[4] [Scikit-learn](https://scikit-learn.org)   
[5] Courses from Master Data Science held at the University of Lille, France