# MachineLearningAlgos
Personal reimplementation of some ML algorithm for learning purposes

## Contribution/ Author

This repo is only made by myself ([Maxence Giraud](https://github.com/MaxenceGiraud/)), I do not seek additional contributors (although you can open an issue if you find a mistake or have a constructive comment) as it has only learning purposes and no other intentions.

If you want to see the work to come and in progress, you can go to the [project tab](https://github.com/MaxenceGiraud/MachineLearningAlgos/projects/1).


## Requirements 
* [NumPy](https://numpy.org/) 
* [SciPy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/) (for plots only)
* [Gym](https://gym.openai.com/) (RL environment, not directly used in the algorithms)

The requirements can be installed through this simple command :
```bash
pip install -r requirements.txt
```

Some readme have written LaTeX, you can view them locally with some capable reader (e.g. VSCode) or using extensions (e.g. [here](https://addons.mozilla.org/en-US/firefox/addon/latexmathifygithub/) for Firefox).

## Installation

To install, simply clone the project (don't forget the requirements first) :
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

To run unittest : 
```bash
python -m unittest tests/*/*_test.py
```

## Implemented Algorithms

Some explainations of the algorithms can be found in the readme of their folders, for more thorough reviews check the references.    
Algorithms that I plan to implement are written in the projects tab.

### [Machine Leaning](./mla/ml)
- [x] [KNN](./mla/ml/knn.py) (classifier+regressor)
- [x] [Logistic Regression/ Perceptron](./mla/ml/logistic_regression.py)
- [x] [Least squares](./mla/ml/leastsquares.py) linear regression/classification
- [x] [Least squares](./mla/ml/leastsquares.py) polynomial Regression/classification
- [x] [Ridge](./mla/ml/leastsquares_regularised.py) 
- [x] [SVM](./mla/ml/svm.py)
- [x] [CART Decision Tree](./mla/ml/decison_tree.py)
- [x] [Gaussian process Regressor](./mla/ml/gaussian_process.py)
- [x] [Naive Bayes](./mla/ml/naive_bayes.py) (Bernoulli, Gaussian and Multinomial)
- [x] [LDA/QDA](./mla/ml/discriminantanalysis.py)
- [x] [Kernel Ridge](./mla/ml/kernel_ridge.py)
- [x] [Kernel KNN](./mla/ml/kernel_knn.py)


### [Ensemble methods](./mla/ensemble/)
- [x] [Random Forest](./mla/ensemble/random_forest.py)
- [x] [Bagging](./mla/ensemble/bagging.py)
- [x] [AdaBoost](./mla/ensemble/adaboost.py)
- [x] [Gradient Boosting](./mla/ensemble/gradient_boosting.py)
- [x] [One vs rest classifier](./mla/ensemble/multiclass.py)
- [x] [One vs one classifier ](./mla/ensemble/multiclass.py)
  

### [Clustering / Unsupervized Learning](./mla/clustering/)
- [x] [K-means](./mla/clustering/kmeans.py)
- [x] [K-medoids](./mla/clustering/kmedoids.py)
- [x] [DBSCAN](./mla/clustering/dbscan.py)
- [x] [OPTICS](./mla/clustering/optics.py)
- [x] [Gaussian Mixture](./mla/clustering/mixture_model.py) with EM algorithm
- [x] [Spectral Clustering](./mla/clustering/spectral_clustering.py)
- [x] [Hierarchical Clustering](./mla/clustering/hierarchical_clustering.py)
- [x] [Fuzzy C-means](./mla/clustering/fuzzycmeans.py)
- [x] [Kernel K-medoids](./mla/clustering/kernel_kmedoids.py)

###  Semi Supzevized Learning
- [x] [Label Propagation](./mla/semi/label_propagration.py)
- [x] [Label Spreading](./mla/semi/label_spreading.py)


### [Deep Leaning / Neural Networks](./mla/dl/)
- [x] [Neural Network Base/sequential](./mla/dl/neuralnetwork.py)

#### [Layers](./mla/dl/layers/)
- [x] [Dense](./mla/dl/layers/dense.py)
- [ ] [Convolution](./mla/dl/layers/convolution.py) (1d,2d)
- [ ] Max/Avg/Min [Pooling](./mla/dl/layers/pooling.py) (1d,2d)
- [x] [Flatten](./mla/dl/layers/flatten.py)
- [x] [Reshape](./mla/dl/layers/reshape.py)
- [x] [Dropout](./mla/dl/layers/dropout.py)

#### Wrappers
- [x] [Autoencoder](./mla/dl/autoencoder.py)
- [x] [MLP](./mla/dl/mlp.py)
  
#### [Optimizers](./mla/dl/optimizer/)
- [x] [Stochastic gradient descent](./mla/dl/optimizer/gradientdescent.py)
- [x] [Mini batch Gradient descent](./mla/dl/optimizer/gradientdescent.py)
- [x] [Epsilon-Delta private SGD](./mla/dl/optimizer/private_sgd.py)
- [x] [Adam](./mla/dl/optimizer/adam.py)
- [x] [Nadam](./mla/dl/optimizer/nadam.py) (Nesterov Adam)
- [x] [Adagrad](./mla/dl/optimizer/adagrad.py)

#### [Loss](./mla/dl/layers/loss/)

All the following loss support weights (if given to fit method, it is done automatically)

- [x] [MSE](./mla/dl/loss/mse.py)
- [x] [MAE](./mla/dl/loss/mae.py)
- [x] [Binary Cross Entropy](./mla/dl/loss/binary_cross_entropy.py)
- [x] [Neg log likelihood](./mla/dl/loss/nll.py)  
  
#### [Activation functions](./mla/dl/activation/)
- [x] [Linear](./mla/dl/activation/linear.py)
- [x] [Relu](./mla/dl/activation/relu.py)
- [x] [Tanh](./mla/dl/activation/tanh.py)
- [x] [Sigmoid](./mla/dl/activation/sigmoid.py)
- [x] [Leaky Relu](./mla/dl/activation/leaky_relu.py)
- [x] [Softplus](./mla/dl/activation/softplus.py)
- [x] [Softmax](./mla/dl/activation/softmax.py)

### [Multi-armed bandit](./mla/mab/)
Credits to [Emilie Kaufmann](http://chercheurs.lille.inria.fr/ekaufman/index.html) for the bandit environment. 

- [x] [UCB](./mla/mab/ucb.py)
- [x] [kl UCB](./mla/mab/klucb.py)
- [x] [Explore Then Commit](./mla/mab/etc.py)
- [x] [Follow the Leader (FTL)](./mla/mab/ftl.py)
- [x] [Thompson sampling](./mla/mab/thompson_sampling.py)
- [x] [Linear UCB](./mla/mab/linear_ucb.py)
- [x] [Linear Thompson sampling](./mla/mab/linear_thompson_sampling.py)
  
For non stationary bandits, see my other repo [here](https://github.com/MaxenceGiraud/ucb-nonstationary).

### [Reinforcement Learning ](./mla/rl/)
- [x] [Value iteration](./mla/rl/value_iteration.py)
- [x] [Q-learning](./mla/rl/qlearning.py)
- [x] [Deep Q-learning](./mla/rl/deep_qlearning.py)
- [x] [Advantage Actor Critic](./mla/rl/advantage_actor_critic.py) (Need PyTorch)

### [Decomposition / Dimensionality reduction](./mla/dimension_reduction/)

- [x] [PCA](./mla/dimension_reduction/pca.py) (for Probabilistic/Bayesian PCA view other repo [here](https://github.com/MaxenceGiraud/BayesianPCA))
- [x] [Kernel PCA](./mla/dimension_reduction/kernel_pca.py) 

### [Kernels](./mla/kernels/) 
*A KernelFusion is also available that allows to combine kernels w.r.t. differents features of the data*

- [x] [RBF](./mla/kernels/rbf.py)
- [x] [Rational Quadratic](./mla/kernels/rational_quadratic.py)
- [x] [Exp Sin Squared](./mla/kernels/expsinesquared.py)
- [x] [Polynomial](./mla/kernels/polynomial.py)
- [x] [Laplacian](./mla/kernels/laplacian.py)
- [x] [Chi Squared](./mla/kernels/chi2.py)
- [x] [Sigmoid](./mla/kernels/sigmoid.py)
- [x] [Linear](./mla/kernels/linear.py)
- [x] [Cosine Similarity](./mla/kernels/cosine_similarity.py)
- [x] [Matérn](./mla/kernels/matern.py)
- [x] (Normalized) [Intersection Kernel](./mla/kernels/intersection.py) (between categorical data)

### [Metrics](./mla/metrics/)
#### [Regression](./mla/metrics/regression/)
- [x] MSE
- [x] RMSE
- [x] MAE
- [x] Median Absolute error
- [x] R2 score
  
#### [Classifcation](./mla/metrics/classification/)
- [x] Accuracy score
- [x] Zero one loss


## References

*The references of the algorithms themselves are for the most part written in the docstring of the corresponding class*

[1] M. BISHOP, Christopher. [Pattern Recognition and Machine Learning.](https://www.springer.com/gp/book/9780387310732) Springer, 2011.    
[2] TIBSHIRANI, Robert, HASTIE, Trevor y FRIEDMAN JEROME, . [The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition.](https://web.stanford.edu/~hastie/ElemStatLearn/) Springer, 2016.   
[3] P. MURPHY, Kevin. [Machine Learning: A Probabilistic Perspective.](https://www.cs.ubc.ca/~murphyk/MLbook/pml-toc-1may12.pdf) The MIT Press, 2012.   
[4] [Scikit-learn](https://scikit-learn.org/stable): Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.   
[5] Courses from [Master Data Science](https://sciences-technologies.univ-lille.fr/mathematiques/formation/master-mention-sciences-des-donnees/) held at the University of Lille, France  