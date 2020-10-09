'''This file contains examples of use for some algo'''
#%%
import mla
import numpy as np

from sklearn.datasets import load_boston,make_classification,fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
#%%
############# Regression task #################

regressor  = mla.LinearRegression() # Choose whatever regressor you want (need to have fit and score methods)

data = load_boston()
X = data['data']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

regressor.fit(X_train,y_train)
regressor.score(X_test,y_test)


#%%
############# Classification task #################

clf = mla.KNN(3) # Choose whatever classifier you want (need to have fit and score methods)

X,y = make_classification(n_samples=300)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf.fit(X_train,y_train)
clf.score(X_test,y_test)


#%%
############# Boolean features Classification task  (Bernoulli Naive Bayes) #################


''' Careful, very big dataset (13000 datapoints with 130 000 features), if low memory use only a fraction of it '''

# Load dataset
X = fetch_20newsgroups()
X,y = X['data'],X['target']

# Convert list of strings to boolean features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)
X = X.toarray()

# for new data to convert :
#X_new = vectorizer.transform(data_new)

X_train,X_test,y_train,y_test = train_test_split(X,y)

# Transform all non zeros to 1
X[np.where(X>1)] =1

bnb = mla.BernoulliNaiveBayes()
bnb.fit(X_train,y_train)
bnb.score(X_test,y_test)
#%%