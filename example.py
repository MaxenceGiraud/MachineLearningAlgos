'''This file contains examples of use for some algo'''
#%%
import mla
from mla import dl,mab,ensemble
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston,make_classification,fetch_20newsgroups,load_iris,fetch_openml,make_blobs
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#%%
############# Regression task #################

regressor  = mla.DecisionTreeRegressor(max_depth=2) # Choose whatever regressor you want (need to have fit and score methods)

data = load_boston()
X = data['data']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

regressor.fit(X_train,y_train)
regressor.score(X_test,y_test)


#%%
############# Binary Classification task #############

clf = mla.KNN(3)# Choose whatever classifier you want (need to have fit and score methods)

X,y = make_classification(n_samples=300)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf.fit(X_train,y_train)
clf.score(X_test,y_test)

#%% 
#############  Multilabel classification task ##########
clf  = ensemble.OneVsOneClassifier(mla.PolynomialClassification) # Choose whatever regressor you want (need to have fit and score methods)

X,y = make_classification(n_classes=4,n_samples=600,n_features=10,n_informative=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf.fit(X_train,y_train)
clf.score(X_test,y_test)
#%%
############# Boolean/Counting features Classification task  (Bernoulli/Multinomial Naive Bayes) #################

''' Careful, very big dataset (13000 datapoints with 130 000 features), if low memory use only a fraction of it '''

# Load dataset
X = fetch_20newsgroups()
X,y = X['data'][:2000],X['target'][:2000]

# Convert list of strings to boolean features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
X = X.toarray()

# for new data to convert :
#X_new = vectorizer.transform(data_new)

X_train,X_test,y_train,y_test = train_test_split(X,y)

#%%
mnb = mla.MultinomialNaiveBayes()
mnb.fit(X_train,y_train)

mnb.score(X_test,y_test)

#%% Boolean 
# Transform all non zeros to 1
X[np.where(X>1)] =1

bnb = mla.BernoulliNaiveBayes()
bnb.fit(X_train,y_train)
bnb.score(X_test,y_test)
#%%
########################### Deep learning ################

## Binary Classification 

# np.random.seed(42)
X, y = make_classification(n_samples=200, n_features=5, n_classes=2, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#%%

nn = dl.NeuralNetwork(X.shape[1:],loss=dl.loss.BinaryCrossEntropy())
nn.add(dl.Dense(10,activation=mla.dl.activation.Sigmoid()))
nn.add(dl.Dense(4,activation=mla.dl.activation.Sigmoid()))
nn.add(dl.Dense(3,activation=mla.dl.activation.Sigmoid()))
nn.add(dl.Dense(1,activation=mla.dl.activation.Sigmoid()))

#%%

nn.fit(X_train,y_train,dl.optimizer.Adam())

print('\n ------------------------ \naccuracy = ',1 - np.count_nonzero(y_test-np.where(nn.predict(X_test)>0.5,1,0).flatten())/y_test.size)
#%%
nn.summary()
#%%
nn.display()

#%% ## Regression

data = load_boston()
X = data['data']
y = data['target']
X = (X - X.min(axis=0))/X.max(axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#%%

nn = dl.NeuralNetwork(X.shape[1:],loss=dl.loss.MAE())
nn.add(dl.Dense(10,activation=mla.dl.activation.Relu()))
nn.add(dl.Dense(4,activation=mla.dl.activation.Relu()))
nn.add(dl.Dense(1))

#%%

nn.fit(X,y,dl.optimizer.Adam(learning_rate=0.1,n_iter=30))

#%% ######### Autoencoder ############################
## Warning : Big dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  
#%%
ae = dl.AutoEncoder(X.shape[1:],loss=dl.MAE())
ae.add(dl.Dense(350))

# ae.add(dl.Dense(50))
ae.add(dl.Dense(150))
ae.add(dl.Dense(100))
ae.add(dl.Dense(20),encoding_layer=True)
#%%
ae.fit(X_train,X_train,dl.Adam(learning_rate=0.01,n_iter=200))

#%%
################# Multi Armed Bandit ##########################
arms = [mab.arms.Bernoulli(0.8),mab.arms.Exponential(2),mab.arms.Gaussian(2.4),mab.arms.Gaussian(1.5)]
bandit = mab.MAB(arms)

T = 100
ucb = mab.UCB(bandit.nbArms)
ts = mab.ThompsonSampling(bandit.nbArms)
etc = mab.ETC(bandit.nbArms,T)

## Compare the different algorithms
mab.bandit_env.RunExpes([ucb,ts,etc],bandit,N_exp=10,timeHorizon=T)
# %%

#%% ##### Gaussian Process Regressor  #################""
import seaborn as sns 

d = 1 #dimension
n = 500 # n samples

x = np.linspace(start=0, stop=1, num=n)
def f(x):
    f = np.sin((4*np.pi)*x) + np.sin((7*np.pi)*x)
    return(f)
sigma_n = 0.4
y = f(x) + np.random.normal(loc=0, scale=sigma_n, size=n)

gp = mla.GaussianProcessRegressor()
gp.fit(x.reshape(-1,1),y)

yh = gp.predict(x,n_samples=100)

# Compare true function to learned one (with noise)
for yi in yh :
    sns.lineplot(x,yi,color='blue',alpha=0.2)
sns.lineplot(x,f(x),color='red',label='True Function')
sns.lineplot(x,y,color='green',label='Training Data',alpha=0.4)
plt.legend()
plt.show()
# %%

###################### Unsupervized Learning ###############

X,_ = make_blobs(600,2,centers=4)

gmm = mla.GaussianMixtureModel(4)
gmm.fit(X)
gmm.display(X) # Only available for GMMS and 2D data

#%%

################# Kernels ##############################

k = mla.kernels.RBF()
k2 = mla.kernels.RBF(l=0.2)
k3 = mla.kernels.RQK()

# Combine kernels with other kernels and other operations
new_kernel= 2**((np.exp(k+k2) / 2) * (abs(k3)** 4) )

print(new_kernel(1,2))
# Same as : 
print(2**(((np.exp(k(1,2)+k2(1,2))) / 2) * abs(k3(1,2))** 4) )

# You can use most of trigonometrics/hyperbolic numpy function to call the kernel as well as exp and log
np.cosh(k)
# you can also apply custom functions :
def custom_f(a):
    return a+2
new_kernel2 = k.apply_func(custom_f)

# And now you can use this new kernel in any kernel algorithm
# %%
