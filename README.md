# MachineLearningAlgos
Personal reimplementation of some ML algorithm for learning purposes

## Author
[Maxence Giraud](https://github.com/MaxenceGiraud/)

## How to
Some examples on how to use the algorithms can be found in the file example.py

## Algoritms

### [Machine Leaning](./mla/ml)
- [x] [KNN](./mla/ml/knn.py) (classifier+regressor)
- [x] [Logistic Regression/ Perceptron](./mla/ml/perceptron.py)
- [x] [Least squares](./mla/ml/leastsquares.py) linear regression/classification
- [x] [Least squares](./mla/ml/leastsquares.py) polynomial Regression/classification
- [x] [Ridge](./mla/ml/leastsquares-_regularised.py) 
- [ ] [LASSO](./mla/ml/leastsquares-_regularised.py) (using Least Angle Regression)
- [ ] [SVM](./mla/ml/svm.py)
- [x] [CART Decision Tree](./mla/ml/decison_tree.py)
- [ ] Gaussian process regression
- [ ] Hidden Markov model
- [ ] Multivariate adaptive regression spline

#### [Bayesian](./mla/ml)
- [x] [Naive Bayes](./mla/ml/naive_bayes.py) (Bernoulli, Gaussian and Multinomial)
- [x] [LDA/QDA](./mla/ml/discriminantanalysis.py)
- [ ] Variational Bayes - Laplace approx (Murphy 8.4.1-2)
- [ ] MCMC
- [ ] Importance sampling
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
- [ ] [Neural Network Base/sequential](./mla/dl/neuralnetwork.py)

#### [Layers](./mla/dl/layers/)
- [ ] [Dense](./mla/dl/layers/dense.py)
- [ ] Convolution (1d,2d,3d)
- [ ] Recurrent
- [ ] LSTM
- [ ] Pooling
- [ ] Dropout
- [ ] Normalization

#### Wrapper
- [ ] Autoencoder/VAE ? (Partially done in class), to refactor as wrapper of NeuralNetwork
- [ ] GAN 
- [ ] Transformer 
  
#### [Optimization Algorithm](./mla/dl/optimizer/)
- [x] Mini batch/full batch Gradient descent
- [x] Stochastic gradient descent
- [ ] Adam
- [ ] Adagrad

#### [Loss](./mla/dl/layers/loss.py)
- [x] MSE
- [x] RMSE
- [x] MAE
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

### Multi-armed bandit
Algo seen in class to add and generalize:   
- [ ] UCB
- [ ] kl UCB
- [ ] ETC 
- [ ] FTL
- [ ] Thompson sampling

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
[3] [Scikit-learn](https://scikit-learn.org)   
[4] Courses from Master Data Science held at the University of Lille, France