import multiprocessing as mp
from functools import partial
import numpy as np
from ..base import BaseClassifier,BaseRegressor
from ..ml.decison_tree import DecisionTreeClassifier,DecisionTreeRegressor

base_learner_reg = partial(DecisionTreeRegressor,3)
base_learner_clf = partial(DecisionTreeClassifier,3)

class BaseAdaBoost:
    def __init__(self,base_model,basemodel_params,n_estimators):
        self.base_model = base_model
        self.basemodel_params = basemodel_params
        self.n_estimators = n_estimators

    def fit(self,X,y):

        # Init weights
        weights = (np.ones(X.shape[0])/X.shape[0]).reshape((-1,1))
        
        # Create base estimators
        self.estimators = []
        self.estimators_tmp = []
        [self.estimators_tmp.append(self.base_model(**self.basemodel_params)) for _ in range(self.n_estimators)]

        # Fit the estimators
        for i in range(self.n_estimators):
            self.estimators_tmp[i].fit(weights*X,y)
            self.estimators.append(self.estimators_tmp[i])
            y_diff = np.abs(self.predict(X).reshape(y.shape)-y)/y # Precitions - true targets

            # Update weights
            weights *= np.exp(y_diff.reshape(-1,1))
           

    def predict_learners(self,X):
        return [self.estimators[i].predict(X) for i in range(len(self.estimators))]

    def predict(self,*args,**kwargs):
        raise NotImplementedError

class AdaBoostClassifier(BaseAdaBoost,BaseClassifier):
    ''' AdaBoost Classifer 
    Ref : Y. Freund, and R. Schapire, “A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting”, 1997.

    Parameters
    ----------
    base_model : Classifer class,
        Base estimator, best to use with weak learners. Default to a Decision Tree with max_depth = 3.
    basemodel_params : dict,
        Parameters of the base estimators,
    n_estimators : int,
        Number of base estimators
    '''
    def __init__(self,base_model = base_learner_clf,basemodel_params = {}, n_estimators = 20):
        super().__init__(base_model=base_model,basemodel_params=basemodel_params,n_estimators=n_estimators)

    def predict(self,X):
        res = super().predict_learners(X)

        # Take the most common value
        res = np.array(res)
        decision = []
        for col in range(res.shape[1]):
            values,counts = np.unique(res[:,col],return_counts=True)
            decision.append(values[counts.argmax()])
        return decision


class AdaBoostRegressor(BaseAdaBoost,BaseRegressor):
    ''' AdaBoost Regressor 
    Ref : Y. Freund, and R. Schapire, “A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting”, 1997.

    Parameters
    ----------
    base_model : Regressor class,
        Base estimator, best to use with weak learners. Default to a Decision Tree with max_depth = 3.
    basemodel_params : dict,
        Parameters of the base estimators,
    n_estimators : int,
        Number of base estimators
    '''
    def __init__(self,base_model = base_learner_reg,basemodel_params = {}, n_estimators = 20):
        super().__init__(base_model=base_model,basemodel_params=basemodel_params,n_estimators=n_estimators)
    
    def predict(self,X):
        res = self.predict_learners(X)

        return np.mean(res,axis=0)