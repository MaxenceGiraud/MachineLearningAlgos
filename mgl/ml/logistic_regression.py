import numpy as np

def sgm(x):
  return 1 / (1 + np.exp(-x))


def batch_gradient_descent(X_plus,y_train,lr,theta0):

    thetas = [theta0]
    nb_iter = 0
    g = 10
    while  np.linalg.norm(g) > 1e-6 and nb_iter < 200 : 
        pred =  sgm(X_plus.T @  thetas[-1])
        err = pred - y_train
        g = (X_plus @ err)#/y_train.shape[0]
        thetas.append(np.array(thetas[-1]-lr*g))
        nb_iter += 1
    return thetas,iter

def stochastic_gradient_descent(X_plus,y_train,lr,theta0):
    
    thetas = [theta0]
    nb_iter = 0
    g=1

    while np.linalg.norm(g) > 1e-6 and nb_iter < 100000 :

        sigma = np.random.permutation
        X_plus = sigma(X_plus)
        y_train = sigma(y_train)

        for i in range(len(y_train)):
            pred =  sgm(X_plus[:,i].T @  thetas[-1])
            err = pred - y_train[i]
            g = X_plus[:,i] * err
            thetas.append(np.array(thetas[-1]-lr*g))
            nb_iter += 1
    return thetas,nb_iter