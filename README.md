# MachineLearningAlgos
Personal reimplementation of some ML algorithm for learning purposes

## Author
[Maxence Giraud](https://github.com/MaxenceGiraud/)

## Requirements 
* [NumPy](https://numpy.org/) 
* [SciPy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/) (for plots only)
* [Gym](https://gym.openai.com/) (RL environment, not directly used in the algorithms)

Some readme have written LaTeX, you can view them locally with some capable reader (e.g. VSCode) or using extensions (e.g. [here](https://addons.mozilla.org/en-US/firefox/addon/latexmathifygithub/) for Firefox).

## Installation

To install, simply clone the project :
```bash
git clone https://github.com/MaxenceGiraud/MachineLearningAlgos
cd MachineLearningAlgos/
```

## Usage

* The machine learning algorithms are programmed in a similar fashion as sklearn.    
* The deep learning framework is conceived in a Keras-like manner.
* The Reinforcement Learning take as input gym-like environments.

```python
import mla
X,y =  load_your_data()

clf = QDA()
clf.fit(X,y)
clf.score(X,y)

##### Unsupevized learning
clust = mla.DBSCAN(eps=5)
clust.fit_predict(X)

##### Deep Learning
from mla import dl

nn = dl.NeuralNetwork(X.shape[1:],loss=dl.MSE())
nn.add(dl.Dense(10,activation=dl.activation.Relu()))
nn.add(dl.Dense(4,activation=dl.activation.Relu()))
nn.add(dl.Dense(1))

nn.fit(X,y)

##### Multi-armed bandits
from mla import mab 

arms = [mab.arms.Bernoulli(0.8),mab.arms.Exponential(2),mab.arms.Gaussian(2.4),mab.arms.Gaussian(1.5)]
bandit = mab.MAB(arms)

T = 100 # Number of trials
ucb = mab.UCB(bandit.nbArms)
ts = mab.ThompsonSampling(bandit.nbArms)
etc = mab.ETC(bandit.nbArms,T)

mab.bandit_env.RunExpes([ucb,ts,etc],bandit,N_exp=10,timeHorizon=T) # Compare the different algorithms
```

Other examples on how to use the algorithms can be found in the file [example.py](./example.py)

## Algorithms

Some explainations of the algorithms can be found in the readme of their folders, for more thorough reviews check the references.

### [Machine Leaning](./mla/ml)
- [x] [KNN](./mla/ml/knn.py) (classifier+regressor)
- [x] [Logistic Regression/ Perceptron](./mla/ml/logistic_regression.py)
- [x] [Least squares](./mla/ml/leastsquares.py) linear regression/classification
- [x] [Least squares](./mla/ml/leastsquares.py) polynomial Regression/classification
- [x] [Ridge](./mla/ml/leastsquares_regularised.py) 
- [ ] [LASSO](./mla/ml/leastsquares_regularised.py) (using Least Angle Regression)
- [ ] Elastic net solver ?? (solver for LS, Ridge, LASSO)
- [ ] SVM
- [x] [CART Decision Tree](./mla/ml/decison_tree.py)
- [x] [Gaussian process Regressor](./mla/ml/gaussian_process.py)
- [ ] MARS (Multivariate adaptive regression spline)
- [x] [Naive Bayes](./mla/ml/naive_bayes.py) (Bernoulli, Gaussian and Multinomial)
- [x] [LDA/QDA](./mla/ml/discriminantanalysis.py)
- [ ] Kernel Ridge


### [Ensemble methods](./mla/ensemble/)
- [x] [Random Forest](./mla/ensemble/random_forest.py)
- [x] [Bagging](./mla/ensemble/bagging.py)
- [x] [AdaBoost](./mla/ensemble/adaboost.py)
- [ ] Gradient Boosting
- [ ] XGBoost
- [x] [One vs rest classifier](./mla/ensemble/multiclass.py)
- [x] [One vs one classifier ](./mla/ensemble/multiclass.py)
  

### [Clustering / Unsupervized Learning](./mla/clustering/)
- [x] [K-means](./mla/clustering/kmeans.py)
- [x] [K-medoids](./mla/clustering/kmedoids.py)
- [x] [DBSCAN](./mla/clustering/dbscan.py)
- [x] [OPTICS](./mla/clustering/optics.py)
- [x] [Gaussian Mixture](./mla/clustering/mixture_model.py) with EM algorithm
- [ ] Variational Bayesian Gaussian Mixture
- [ ] Generative topographic map
- [ ] Spectral Clustering
- [x] [Fuzzy C-means](./mla/clustering/fuzzycmeans.py)
  
### Other ML/Ensemble methods
- [ ] Isolation Forest (detection of outliers)  

### [Deep Leaning](./mla/dl/)
- [x] [Neural Network Base/sequential](./mla/dl/neuralnetwork.py)

#### [Layers](./mla/dl/layers/)
- [x] [Dense](./mla/dl/layers/dense.py)
- [ ] Convolution (1d,2d,3d)
- [ ] Recurrent
- [ ] LSTM
- [ ] GRU
- [ ] Max/Avg/Min Pooling (1d,2d,3d)
- [ ] Deconvolution/ Upconv / Transposed Conv. layer
- [x] Flatten
- [x] Reshape
- [x] Dropout

### Blocks
- [ ] Transformer
- [ ] Inception
- [ ] Residual (ResNet)

#### Wrapper
- [x] [Autoencoder](./mla/dl/autoencoder.py)
- [x] [MLP](./mla/dl/mlp.py)
- [ ] VAE
- [ ] GAN  
  
#### [Optimization Algorithm](./mla/dl/optimizer/)
- [x] Stochastic gradient descent
- [x] Mini batch Gradient descent
- [x] Epsilon-Delta private SGD
- [x] Adam
- [ ] Nadam (Nesterov Adam)
- [x] Adagrad
- [ ] L-BFGS

#### [Loss](./mla/dl/layers/loss.py)
- [x] MSE
- [x] MAE
- [x] Binary Cross Entropy
- [ ] Neg log likelihood  
- [x] Add possibility of weighted loss
  
#### [Activation functions](./mla/dl/activation/activation.py)
- [x] Linear
- [x] Relu
- [x] Tanh
- [x] Sigmoid
- [x] Leaky Relu
- [x] Softplus
- [x] Softmax

### [Multi-armed bandit](./mla/mab/)
Credits to [Emilie Kaufmann](http://chercheurs.lille.inria.fr/ekaufman/index.html) for the bandit environment. 

- [x] UCB
- [x] kl UCB
- [x] ETC 
- [x] FTL
- [x] Thompson sampling
- [x] Linear UCB
- [x] Linear Thompson sampling
  
For non stationary bandits, see my other repo [here](https://github.com/MaxenceGiraud/ucb-nonstationary).

### [Reinforcement Learning ](./mla/rl/)
- [x] [Value iteration](./mla/rl/value_iteration.py)
- [x] [Q-learning](./mla/rl/qlearning.py)
- [ ] Deep Q-learning
- [ ] Advantage Actor Critic

### Decomposition / Dimensionality reduction

- [x] PCA (for Probabilistic,Bayesian PCA and Mixture of PCA view other repo [here](https://github.com/MaxenceGiraud/BayesianPCA))
- [ ] Kernel PCA
- [ ] FDA
- [ ] ICA
- [ ] Elastic map
- [ ] CCA
- [ ] Kernel CCA

#### Manifold

- [ ] Multi-dimensional Scaling
- [ ] Isomap Embedding
- [ ] Spectral Embedding

### Kernels 

- [x] [RBF](./mla/kernels/rbf.py)


# TODO / Ongoing dev
- [ ] Implement backprop of Convolution layer
- [ ] Implement backprop of 1D Pooling Layers
- [ ] Find out how OPTICS makes the clusters from reachability graph
- [ ] Implement LASSO using Least angle regression
- [ ] Add Deep Q learning and Advantage Actor Critic algos
- [ ] Add Explanations of all currently implemented algos

## References


[1] M. BISHOP, Christopher. [Pattern Recognition and Machine Learning.](https://www.springer.com/gp/book/9780387310732) Springer, 2011.    
[2] TIBSHIRANI, Robert, HASTIE, Trevor y FRIEDMAN JEROME, . [The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition.](https://web.stanford.edu/~hastie/ElemStatLearn/) Springer, 2016.   
[3] P. MURPHY, Kevin. [Machine Learning: A Probabilistic Perspective.](https://www.cs.ubc.ca/~murphyk/MLbook/pml-toc-1may12.pdf) The MIT Press, 2012.   
[4] [Scikit-learn](https://scikit-learn.org/stable): Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.   
[5] Courses from [Master Data Science](https://sciences-technologies.univ-lille.fr/mathematiques/formation/master-mention-sciences-des-donnees/) held at the University of Lille, France