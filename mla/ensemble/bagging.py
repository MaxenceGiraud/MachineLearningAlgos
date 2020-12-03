import multiprocessing as mp
from functools import partial
import numpy as np
from ..base import BaseClassifier,BaseRegressor
from ..ml.decison_tree import DecisionTreeClassifier,DecisionTreeRegressor

class BaseBagging:
    def __init__(self,base_model,basemodel_params={},n_estimators = 20,samples_ratio=0.8,features_ratio=0.5,parallelize=False):
        self.base_model = base_model
        self.basemodel_params = basemodel_params
        self.parallelize = parallelize
        self.n_estimators = n_estimators
        self.samples_ratio = samples_ratio
        self.features_ratio = features_ratio
        
        self.estimators = []

    def fit(self,X,y):
        
        # Draw bootstrap samples
        n_sample = int(X.shape[0] * self.samples_ratio)
        samples_idx = np.random.choice(X.shape[0],size=self.n_estimators*n_sample ,replace=True).reshape(self.n_estimators,n_sample)
        # Draw bootstrap features
        n_feature =  int(X.shape[1] * self.features_ratio)
        self.features_idx  = np.random.choice(X.shape[1],size=self.n_estimators *n_feature ,replace=True).reshape(self.n_estimators,n_feature)
            


        # Create base tree
        self.estimators = []
        [self.estimators.append(self.base_model(**self.basemodel_params)) for _ in range(self.n_estimators)]

        # Fit the trees
        if self.parallelize :
            fit_func = [partial(self.estimators[i].fit,X[samples_idx[i]][:,self.features_idx[i]],y[samples_idx[i]]) for i in range(self.n_estimators)]

            pool = mp.Pool(mp.cpu_count())
            [pool.apply(fit_func[i]) for i in range(self.n_estimators)]

        else : 
            for i in range(self.n_estimators) :
                self.estimators[i].fit(X[samples_idx[i]][:,self.features_idx[i]],y[samples_idx[i]])

    def predict_all_trees(self,X):
        if self.parallelize :
            pool = mp.Pool(mp.cpu_count())
            for i in range(self.n_estimators):
                res = [pool.apply(self.estimators[i].predict,(X[:,self.features_idx[i]],)) for i in range(self.n_estimators)]
        else : 
            res = []
            for i in range(self.n_estimators):
                res.append(self.estimators[i].predict(X[:,self.features_idx[i]]))

        return res

class BagginClassifier(BaseBagging,BaseClassifier):
    ''' Bagging Classfier
    Parameters
    ----------
    basetree : object,
            decision tree classifier object with fit and predict methods
            (may be also another base estimator)
    basetree_params : dict,
            parameters of base tree
    parallelize : bool, 
    n_estimators : int,
            number of trees in the forest
    '''
    def __init__(self,base_model=DecisionTreeClassifier,basemodel_params={},n_estimators = 20,samples_ratio=0.8,features_ratio=0.5,parallelize=True):
        super().__init__(base_model=base_model,basemodel_params=basemodel_params,n_estimators=n_estimators,samples_ratio=samples_ratio,features_ratio=features_ratio,parallelize=parallelize)

    def fit(self,X,y):
        self.labels = np.unique(y)

        return self.fit(X,y)
        
    def predict(self,X):
        res = super().predict_all_trees(X)

        # Take the most common value
        res = np.array(res)
        decision = []
        for col in range(res.shape[1]):
            decision.append(np.bincount(res[:,col]).argmax())
        return decision

class BaggingRegressor(BaseBagging,BaseRegressor):
    ''' Bagging Classfier
    Parameters
    ----------
    basetree : object,
            decision tree regressor object with fit and predict methods
            (may be also another base estimator)
    basetree_params : dict,
            parameters of base tree
    parallelize : bool, 
    n_estimators : int,
            number of trees in the forest
    '''
    def __init__(self,base_model=DecisionTreeRegressor,basemodel_params={},n_estimators = 20,samples_ratio=0.8,features_ratio=0.5,parallelize=True):
        super().__init__(base_model=base_model,basemodel_params=basemodel_params,n_estimators=n_estimators,samples_ratio=samples_ratio,features_ratio=features_ratio,parallelize=parallelize)

    
    def predict(self,X):
        res = self.predict_all_trees(X)

        return np.mean(res,axis=0)