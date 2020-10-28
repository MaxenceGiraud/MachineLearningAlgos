import numpy as np
# import graphviz

class Node:
    def __init__(self,criterion,idx_feature,gini = None,categorial = False):
        self.criterion = criterion
        self.idx_feature = idx_feature
        self.left = None
        self.right = None
        self.gini = gini
        self.categorial = categorial

    @property
    def depth(self):
        return max(self.left.depth,self.right.depth) + 1 
    
    def predict(self,X):
        if self.categorial :
            left,right = np.where(X[:,self.idx_feature] == self.criterion),np.where(X[:,self.idx_feature] != self.criterion) 
        else :
            left,right = np.where(X[:,self.idx_feature] < self.criterion),np.where(X[:,self.idx_feature] >= self.criterion)       
        prediction = np.zeros(X.shape[0])
        prediction[left] = self.left.predict(X[left])
        prediction[right] = self.right.predict(X[right])

        return prediction

    def print(self):
        # TODO display tree
        pass
   
class Leaf(Node):
    ''' A leaf is a node at the bottom of the tree, it takes the decision'''
    def __init__(self,decision):
        self.criterion = None
        self.idx_feature = None
        self.left = None
        self.right = None
        self.decision = decision

    @property
    def depth(self):
        return 0

    def predict(self,X):
        return np.ones(X.shape[0])*self.decision

def gini_index(groups,class_labels):
    ''' Compute Gini index for a given split for classification 
    
    Parameters
    ----------
    groups : list of array,
        groups decided by the split,  array is the list of the true labels 
    
    class_labels : list
        labels given by the split

    Yield
    -----
    gini : float,
        gini index
    '''
    counts  = [len(group) for group in groups]
    n_samples = np.sum(counts)

    gini = 0
    for i in range(len(groups)):
        if counts[i] == 0:
            continue
        score = 0
        for label in class_labels :
            proportion = np.sum(groups[i] == label) / counts[i]
            score += proportion**2
        gini += (1-score) * (counts[i]/n_samples)
    return gini

    

class DecisionTreeClassifier:
    ''' CART Decision tree classifier
    
    Parameters
    ----------
    max_depth :int ,
            maximum depth of the tree         
    min_samples_split : int,
            minimum number of samples in order to create a split
    categorical_features : list,
            list of features which are categorical (index of the column)
     '''

    def __init__(self,max_depth=10,min_samples_split=2,categorial_features = []):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.categorial_features  = categorial_features

    def get_depth(self):
        return self.tree.depth

    def get_best_split(self,X,y):
        ''' get the best one split possible '''
        best_score = 10
        labels  = np.unique(y)
        for j in range(X.shape[1]):
            # iterate on unique value of features
            for i in np.unique(X[:,j],return_index=True)[1]:
                if j in self.categorial_features :
                    groups = np.where(X[:,j]==X[i,j])[0], np.where(X[:,j]!=X[i,j])[0]
                else :
                    groups = np.where(X[:,j]<X[i,j])[0], np.where(X[:,j]>=X[i,j])[0]
                gini = gini_index([[y[groups[0]]][0],[y[groups[1]]][0]],labels)
                if gini < best_score:
                    best_score = gini
                    best_idx = i,j
                    best_groups = groups

        return best_idx,best_score,best_groups

    def build_node(self,X,y,depth):
        
        # Create a leaf (end of the tree)
        if self.max_depth <= depth or self.min_samples_split >= X.shape[0]:
            current_node = Leaf(decision = np.bincount(y).argmax())
            return current_node

        idx,gini,groups_idx = self.get_best_split(X,y) # Get the best split
        
        current_node = Node(X[idx[0],idx[1]],idx[1],gini,categorial= (idx[1] in self.categorial_features))
        if gini == 0 : # TODO May extend property to epsilon > 0
            current_node.left = Leaf(decision = np.bincount(y[groups_idx[0]]).argmax())
            current_node.right = Leaf(decision = np.bincount(y[groups_idx[1]]).argmax())
        else :
            # Build the children nodes
            current_node.left = self.build_node(X[groups_idx[0]],y[groups_idx[0]],depth+1)
            current_node.right = self.build_node(X[groups_idx[1]],y[groups_idx[1]],depth+1)

        return current_node

    def fit(self,X,y):
        self.tree = self.build_node(X,y,0)     
        
    def predict(self,X):
        return self.tree.predict(X)

    def score(self,X,y):
        y_hat  = self.predict(X)
        acc  = np.count_nonzero(np.array(y_hat)==np.array(y)) /len(y)
        return acc

class DecisionTreeRegressor:
    ''' CART Decision tree regressor
   Parameters
    ----------
    max_depth :int ,
            maximum depth of the tree         
    min_samples_split : int,
            minimum number of samples in order to create a split
    categorical_features : list,
            list of features which are categorical (index of the column)
    metric : string,
            metric used to choose the best split
     '''

    def __init__(self,max_depth=10,min_samples_split=2,categorial_features = [],metric = "mse"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.categorial_features  = categorial_features
        self.metric = metric # ignored for now

    def get_depth(self):
        return self.tree.depth

    def get_best_split(self,X,y):
        ''' get the best one split possible '''
        best_score = np.inf
        labels  = np.unique(y)
        for j in range(X.shape[1]):
            # iterate on unique value of features
            for i in np.unique(X[:,j],return_index=True)[1]:
                if j in self.categorial_features :
                    groups = np.where(X[:,j]==X[i,j])[0], np.where(X[:,j]!=X[i,j])[0]
                else :
                    groups = np.where(X[:,j]<X[i,j])[0], np.where(X[:,j]>=X[i,j])[0]
                
                n_samples = np.sum([len(group) for group in groups])
                score = np.sum([np.square(group-np.mean(group)).sum() * len(group)/n_samples for group in groups])   
                if score < best_score:
                    best_score = score
                    best_idx = i,j
                    best_groups = groups

        return best_idx,best_score,best_groups

    def build_node(self,X,y,depth):
        
        # Create a leaf (end of the tree)
        if self.max_depth <= depth or self.min_samples_split >= X.shape[0]:
            current_node = Leaf(decision = np.mean(y))
            return current_node

        idx,score,groups_idx = self.get_best_split(X,y) # Get the best split
        
        current_node = Node(X[idx[0],idx[1]],idx[1],score,categorial= (idx[1] in self.categorial_features))
        if score == 0 : # TODO May extend property to epsilon > 0
            current_node.left = Leaf(decision = np.mean(y[groups_idx[0]]))
            current_node.right = Leaf(decision = np.mean(y[groups_idx[1]]))
        else :
            # Build the children nodes
            current_node.left = self.build_node(X[groups_idx[0]],y[groups_idx[0]],depth+1)
            current_node.right = self.build_node(X[groups_idx[1]],y[groups_idx[1]],depth+1)

        return current_node

    def fit(self,X,y):
        self.tree = self.build_node(X,y,0)     
        
    def predict(self,X):
        return self.tree.predict(X)

    def score(self,X,y):
        '''Compute MSE for the prediction  of the model with X/y'''
        y_hat = self.predict(X)
        mse = np.sum((y - y_hat)**2) / len(y)
        return mse