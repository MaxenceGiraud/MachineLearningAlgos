import numpy as np

def BellmanOperator(Q, env, gamma=0.99):
    """ Apply the bellman operator to Q
    Parameters
    ----------
    Q : array of shape (S,A),
        Action Value function/matrix, shape (S,A) with S the state space dimension and A the action space dimension.
    env : gym-like env,
        environment

    Yields
    -------
    TQ : array of shape (S,A),
        Results of the Bellman operator applied to the Action-Value function/matrix Q
    """
    S = env.observation_space.n
    A = env.action_space.n 
    TQ = np.zeros((S, A))

      for s in range(S):
        for a in range(A):
            TQ[s,a] = env.R[s,a] + gamma * env.P[s,a] @ Q.max(axis=1)

    return TQ


def ValueIteration(env, gamma=0.99, epsilon=1e-6):
    ''' Value iteration,
    return optimal Action space function given the reward r and the transition probabilities P 
    
    Parameters
    -----------
    env : gym-like env,
        environment with know reward and transition probabilities, and finite state and action space
    gamma : float,
        Discount factor ensuring the convergence of the iterations
    epsilon : float,
        Threshold used to stop the iteration

    Yields
    ------
    Q : array of shape (S,A), 
        Optimal action Value function/matrix that is the fixed point of the Bellman operator T , shape (S,A) with S the state space dimension and A the action space dimension.
    '''

    S = env.observation_space.n
    A = env.action_space.n 
    Q = np.zeros((S, A))

    while True : 
        TQ = BellmanOperator(Q,env,gamma)
        if np.abs(TQ - Q).max() <= epsilon :
            break
        Q = TQ

    return Q