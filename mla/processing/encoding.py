import numpy as np
import re

def object_to_categorical(X):
    ''' Convert Data with features encoding as "objects" (such as string/number) into boolean categorical features

    Parameters
    ----------
    X : list of list of objects of lenght n_data

    Yields 
    ------
    X_new : Boolean np array of shape (n_data,n,features)
    features : list of objects
    '''
    features = list(set([ xi for x in X for xi in x])) # List of unique features in X
    X_new = np.zeros((len(X),len(features)))
    for i in range(len(X)):
        X_new[i] = np.where([f in X[i] for f in features],1,0)

    return X_new,features


def object_to_categorical_from_features(X,features):
    ''' Convert Data with features encoding as "objects" (such as string/number) into boolean categorical features using a preexisting list of features '''
    X_new = np.zeros((len(X),len(features)))
    for i in range(len(X)):
        X_new[i] = np.where([f in X[i] for f in features],1,0)

    return X_new


def string_to_wordlist(str):
    ''' Convert a string into list of word (or anything separated by " ")'''
    return re.sub("[^\w]", " ", str).split()