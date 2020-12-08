import numpy as np

def q_learning(env,lr=1,eps = 0.5, gamma=0.99, T=5000):
    """Q Learning Algorithm,
    Parameters 
    ----------
    env : gym-like env,
        environment with finite state and action space
    lr : float/int or function of t and nv(number of visits of the state action couple),
        Learning Rate
    eps : float/int or function of t,
        Exploration parameter,
    gamma : float,
        Discount factor
    T : int,
        Number of iteration

    Yields
    ------
    Q : array of shape (S,A), 
        Action Value function/matrix, shape (S,A) with S the state space dimension and A the action space dimension.
    """
    ## INIT
    S = env.observation_space.n
    A = env.action_space.n 
    Q = np.random.random((S, A))  # How can we improve this initialization?  

    visited_matrix = np.ones((S,A))
    state = env.reset()

    if isinstance(lr,float) or isinstance(lr,int):
        lr = lambda t,nv : lr
    
    if isinstance(eps,float) or isinstance(eps,int):
        eps = lambda t : eps

    
    for t in range(T):
        if np.random.random() < eps(t) : # Exploration
            action = env.action_space.sample()
        else :
            action = Q[state].argmax()

        # Take action and observe
        next_state, reward, is_terminal, _ = env.step(action)

        # Update Q
        delta_t = reward + gamma * Q[next_state].max() - Q[state,action]
        Q[state,action] += lr(t,visited_matrix[state,action])*delta_t

        # Transition to next iteration
        visited_matrix[state,action] += 1
        state = next_state

        # If experiment over, reset the env
        if is_terminal :
            state = env.reset()

    return Q 