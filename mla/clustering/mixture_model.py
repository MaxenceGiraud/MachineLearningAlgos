import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.spatial.distance import cdist
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

class GaussianMixtureModel:
    ''' Gaussian Mixture Model clustering algorithm
    Ref :

    Parameters
    ----------
    n_clusters : int,
        Number of cluster/gaussians to consider
    iter_max : int,
        maximum number of iterations of the algo
    eps : float,
        stopping criterion
    init_means : string,
        Method to initialize means, only kmeans++ and random are supported
    '''
    def __init__(self,n_cluster=3,iter_max=100,eps=1e-3,init_means="kmeans++"):
        self.n_cluster = n_cluster
        self.iter_max = iter_max
        self.eps = eps

        self.init_means = init_means


    def _init_params(self,X):

        # Init means using kmeans++
        if self.init_means == 'kmeans++':
            self.mu = [X[np.random.randint(0,X.shape[0])]]
            for i in range(self.n_cluster-1):
                prob_dist_squared = np.min(cdist(self.mu,X),axis=0)**2
                prob_dist_squared = prob_dist_squared / sum(prob_dist_squared) # Normalize probability
                self.mu = np.vstack([self.mu,X[np.random.choice(X.shape[0],1,p=prob_dist_squared)]]) 
        else :
            # Random init of means
            self.mu = np.random.uniform(low=np.min(X,axis=0), high=np.max(X,axis=0), size=(self.n_cluster,X.shape[1]))  

        # Init Cov matrix
        self.sigma  =np.zeros((self.n_cluster,X.shape[1],X.shape[1]))
        for k in range(self.n_cluster):
            self.sigma[k] = np.cov(X.T)
            min_eig = np.min(np.real(np.linalg.eigvals(self.sigma[k]))) # Make cov PSD
            if min_eig < 0:
                self.sigma[k] -= 10*min_eig * np.eye(*self.sigma[k].shape)
        
        self.pi = np.ones(self.n_cluster)/X.shape[0] # mixing coef
        self.r = np.zeros((X.shape[0],self.n_cluster)) # responsability

    def _compute_resp(self,X):
        r = np.zeros((X.shape[0],self.n_cluster))
        for k in range(self.n_cluster):
            r[:,k] = self.pi[k] *  multivariate_normal.pdf(x=X,mean=self.mu[k],cov=self.sigma[k])
        r /= np.sum(r,axis=1).reshape(-1,1)
        return r

    def _expectation_step(self,X):
        self.r = self._compute_resp(X)

    def _maximization_step(self,X):
        self.N = np.sum(self.r,axis=0)
        for k in range(self.n_cluster):
            # Update Means
            self.mu[k] = (self.r[:,k] @ X) / self.N[k] 

            # Update Covariance Matrix
            self.sigma[k] = np.zeros((X.shape[1],X.shape[1]))
            xmu = X - self.mu[k]
            self.sigma[k] = self.r[:,k].reshape(1,-1) * xmu.T @ xmu
            self.sigma[k] = np.sqrt(np.abs(self.sigma[k]))/ self.N[k] + np.eye(X.shape[1]) * 1e-6
            
        self.pi = self.N / X.shape[0]

    def fit(self,X):
        
        self._init_params(X)

        iter = 0
        nll = np.inf
        old_nll = 0
        while iter < self.iter_max and np.abs(nll-old_nll) > self.eps:
            print(iter)

            # Expectation step           
            self._expectation_step(X)

            # Maximimzation step
            self._maximization_step(X)
            
            old_nll = nll
            nll = 0.0
            for k in range(self.n_cluster):
                nll += self.pi[k] * multivariate_normal.pdf(x=X,mean=self.mu[k],cov=self.sigma[k])
            nll = -np.sum(np.log(nll))
            print("Neg log likelihood =",nll)
            iter +=1
            
    def fit_predict(self,X):
        self.fit(X)
        return np.argmax(self.r,axis=1)

    def predict(self,X):
        r = self._compute_resp(X)
        return np.argmax(r,axis=1)


    def display(self,X):
        assert X.shape[1] == 2 , 'To display X must have 2 features (Only 2D is supported)'
        clusters = self.predict(X)

        fig, ax = plt.subplots()

        ax.scatter(X[:,0],X[:,1],c=clusters) # Data
        ax.scatter(self.mu[:,0],self.mu[:,1],c='black') # Means

        # Draw confidence ellipses
        for i in range(self.n_cluster):
            confidence_ellipse(self.mu[i],self.sigma[i],ax,n_std=1.0,edgecolor="firebrick")
            confidence_ellipse(self.mu[i],self.sigma[i],ax,n_std=2.0,edgecolor="firebrick")
            confidence_ellipse(self.mu[i],self.sigma[i],ax,n_std=3.0,edgecolor="firebrick")
            confidence_ellipse(self.mu[i],self.sigma[i],ax,n_std=5.0,edgecolor="firebrick")
            confidence_ellipse(self.mu[i],self.sigma[i],ax,n_std=7.0,edgecolor="firebrick")

        plt.show()


def confidence_ellipse(means,cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    Based on :  https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html

    Parameters
    ----------
    cov : array-like, shape (d,d)
        Covariance Matrix

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    mean_x,mean_y = means
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)