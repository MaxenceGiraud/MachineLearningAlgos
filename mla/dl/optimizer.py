import numpy as np

def gradient_descent(X, y, theta, gradient, batch_size=32, learning_rate=1, eps=1e-6, iter_max=100):

    # add column of 1 for the bias/intercept
    X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    g = 1
    nb_iter = 0
    while np.linalg.norm(g) > eps and nb_iter < iter_max:
        # Random permutation
        permut = np.random.permutation(X.shape[0])
        X = X[permut]
        y = y[permut]

        for i in range(len(y) // batch_size):
            range_start, range_end = i*batch_size, (i+1)*batch_size

            # Compute gradient
            g = gradient(X[range_start:range_end])

            # Update weights
            theta -= learning_rate*g
        # last mini batch
        g = gradient(X[i*batch_size])
        theta -= learning_rate*g

        nb_iter += 1
    return theta


def adam():
    pass


def adagrad():
    pass
