# MachineLearningAlgos
Personal reimplementation of some ML algorithm for learning purposes

## Author
[Maxence Giraud](https://github.com/MaxenceGiraud/)

## Requirements 
* [NumPy](https://numpy.org/) 
* [Matplotlib](https://matplotlib.org/) (for plots only)
* [Gym](https://gym.openai.com/) (RL environment)

Some readme have written LaTeX, you can view them locally with some capable reader (e.g. VSCode) or using extensions (e.g. [here](https://addons.mozilla.org/en-US/firefox/addon/latexmathifygithub/) for Firefox).

## How to
Some examples on how to use the algorithms can be found in the file [example.py](./example.py)


## Algorithms

Some explainations of the algorithms can be found in the readme of their folders, for more thorough reviews check the references.

### [Machine Leaning](./mla/ml)
- [x] [KNN](./mla/ml/knn.py) (classifier+regressor)
- [x] [Logistic Regression/ Perceptron](./mla/ml/perceptron.py)
- [x] [Least squares](./mla/ml/leastsquares.py) linear regression/classification
- [x] [Least squares](./mla/ml/leastsquares.py) polynomial Regression/classification
- [x] [Ridge](./mla/ml/leastsquares_regularised.py) 
- [ ] [LASSO](./mla/ml/leastsquares_regularised.py) (using Least Angle Regression)
- [ ] Elastic net solver ?? (solver for LS, Ridge, LASSO)
- [ ] [SVM](./mla/ml/svm.py)
- [x] [CART Decision Tree](./mla/ml/decison_tree.py)
- [ ] Gaussian process
- [ ] MARS (Multivariate adaptive regression spline)
- [x] [Naive Bayes](./mla/ml/naive_bayes.py) (Bernoulli, Gaussian and Multinomial)
- [x] [LDA/QDA](./mla/ml/discriminantanalysis.py)
  

### [Ensemble methods](./mla/ensemble/)
- [x] [Random Forest](./mla/ensemble/random_forest.py)
- [x] [Bagging](./mla/ensemble/bagging.py)
- [ ] Gradient Boosting
- [ ] AdaBoost
- [x] [One vs rest classifier](./mla/ensemble/multiclass.py)
- [x] [One vs one classifier ](./mla/ensemble/multiclass.py)
  

### [Clustering / Unsupervized Learning](./mla/clustering/)
- [x] [K-means](./mla/clustering/kmeans.py)
- [x] [K-medoids](./mla/clustering/kmedoids.py)
- [x] [DBSCAN](./mla/clustering/dbscan.py)
- [x] [OPTICS](./mla/clustering/optics.py)
- [ ] [Gaussian Mixture](./mla/clustering/mixture_model.py) with EM algorithm
- [ ] Variational Bayesian Gaussian Mixture
- [ ] Generative topographic map
- [x] [Fuzzy C-means](./mla/clustering/fuzzycmeans.py)
  

### [Deep Leaning](./mla/dl/)
- [x] [Neural Network Base/sequential](./mla/dl/neuralnetwork.py)

#### [Layers](./mla/dl/layers/)
- [x] [Dense](./mla/dl/layers/dense.py)
- [ ] Convolution (1d,2d,3d)
- [ ] Deconvolution
- [ ] Recurrent
- [ ] LSTM
- [ ] GRU
- [ ] Transformer Block
- [ ] Pooling
- [x] Flatten
- [x] Reshape
- [x] Dropout

#### Wrapper
- [x] [Autoencoder](./mla/dl/autoencoder.py)
- [ ] VAE
- [ ] GAN  
  
#### [Optimization Algorithm](./mla/dl/optimizer/)
- [x] Stochastic gradient descent
- [x] Mini batch batch Gradient descent
- [x] <img src="https://render.githubusercontent.com/render/math?math=\epsilon">-<img src="https://render.githubusercontent.com/render/math?math=\delta"> private SGD
- [x] Adam
- [ ] Nadam (Nesterov Adama)
- [x] Adagrad

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

### Reinforcement Learning 
- [ ] Value iteration
- [ ] Q-learning
  
### Data processing/Analysis
- [ ] PCA
- [ ] FDA
- [ ] ICA
- [ ] Elastic map
- [ ] CCA

## References


[1] M. BISHOP, Christopher. [Pattern Recognition and Machine Learning.](https://www.springer.com/gp/book/9780387310732) Springer, 2011.    
[2] TIBSHIRANI, Robert, HASTIE, Trevor y FRIEDMAN JEROME, . [The Elements of Statistical Learning: Data Mining, Inference, and Prediction, Second Edition.](https://web.stanford.edu/~hastie/ElemStatLearn/) Springer, 2016.   
[3] P. MURPHY, Kevin. [Machine Learning: A Probabilistic Perspective.](https://www.cs.ubc.ca/~murphyk/MLbook/pml-toc-1may12.pdf) The MIT Press, 2012.   
[4] [Scikit-learn](https://scikit-learn.org)   
[5] Courses from [Master Data Science](https://sciences-technologies.univ-lille.fr/mathematiques/formation/master-mention-sciences-des-donnees/) held at the University of Lille, France
