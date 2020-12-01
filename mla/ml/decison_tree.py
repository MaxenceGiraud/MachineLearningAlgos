import numpy as np
from ..base import BaseClassifier,BaseRegressor
from copy import deepcopy
# import graphviz

class Node:
    ''' Node of a Decision tree'''
    def __init__(self,criterion,idx_feature,gini = None,categorial = False):
        self.criterion = criterion
        self.idx_feature = idx_feature
        self.left = None
        self.right = None
        self.gini = gini
        self.categorial = categorial
        self.pruned = False

    @property
    def depth(self):
        return max(self.left.depth,self.right.depth) + 1 
    
    def split_data(self,X):
        if self.categorial :
            left,right = np.where(X[:,self.idx_feature] == self.criterion),np.where(X[:,self.idx_feature] != self.criterion) 
        else :
            left,right = np.where(X[:,self.idx_feature] < self.criterion),np.where(X[:,self.idx_feature] >= self.criterion) 

        return left,right 
    
    def predict(self,X):
        left,right = self.split_data(X)
        prediction = np.zeros(X.shape[0])
        prediction[left] = self.left.predict(X[left])
        prediction[right] = self.right.predict(X[right])

        return prediction

    def init_pruned(self):
        self.pruned = False
        self.left.init_pruned()
        self.right.init_pruned()
    

    def print(self,pos=[0]):
        self.left.print(pos.append(0))
        self.right.print(pos.append(1))
   
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
    
    def init_pruned(self):
        pass
    
    def split_data(self,X):
        pass

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

class BaseDecisionTree:
    def __init__(self,max_depth=10,min_samples_split=2,categorial_features = [],pruning_eps=0.05):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.categorial_features  = categorial_features
        self.pruning_eps = pruning_eps
        self.tree = None
    
    @property
    def depth(self):
        return self.tree.depth
    
    def fit(self,X,y):
        self.tree = self.build_node(X,y,0)  
    
    def _create_leaf(self,y):
        raise Exception

    def get_best_split(self,X,y):
        raise Exception
    
    def _create_pruned_tree(self,node,X,y):
        if isinstance(node.left,Leaf) and isinstance(node.right,Leaf) :
            if not node.pruned:
                node.pruned = True
                return True
        elif not isinstance(node,Leaf): 
            left,right = node.split_data(X)
            temp_left = self._create_pruned_tree(node.left,X[left],y[left])
            temp_right = self._create_pruned_tree(node.right,X[right],y[right])
            if temp_left == 'done' :
                return 'done'
            elif temp_left == True :
                node.left = self._create_leaf(y)
                return 'done'

            if temp_right == 'done' :
                return 'done'
            elif temp_right == True :
                node.right = self._create_leaf(y)
                return 'done'
            
        return False
    
    def get_pruned_trees(self,X,y):
        # Get all possible pruned trees  
        tree_list = [self.tree]
        self.tree.init_pruned()  # Init pruned values of nodes
        while True:
            new_tree = deepcopy(tree_list[-1])
            if self._create_pruned_tree(new_tree,X,y) != False:
                tree_list.append(new_tree)
            else : 
                break
        return tree_list

    def _bottom_up_pruning(self,X,y):
        raise NotImplementedError


    def _top_down_pruning(self,X,y):
        raise NotImplementedError
    
    def prune(self,X,y,method='bottom_up'):
        if method == 'bottom_up':
            self._bottom_up_pruning(X,y)
        
        elif method == 'top_down':
            self._top_down_pruning(X,y)  

    
    def build_node(self,X,y,depth):
        # Create a leaf (end of the tree)
        if self.max_depth <= depth or self.min_samples_split >= X.shape[0]:
            current_node = self._create_leaf(y)
            return current_node

        idx,score,groups_idx = self.get_best_split(X,y) # Get the best split
        
        current_node = Node(X[idx[0],idx[1]],idx[1],score,categorial= (idx[1] in self.categorial_features))
        if score == 0 : # TODO May extend property to epsilon > 0
            current_node.left =  self._create_leaf(y[groups_idx[0]])
            current_node.right = self._create_leaf(y[groups_idx[1]])
        else :
            # Build the children nodes
            current_node.left = self.build_node(X[groups_idx[0]],y[groups_idx[0]],depth+1)
            current_node.right = self.build_node(X[groups_idx[1]],y[groups_idx[1]],depth+1)

        return current_node
        
    def predict(self,X):
        return self.tree.predict(X)


class DecisionTreeClassifier(BaseDecisionTree,BaseClassifier):
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
                gini = gini_index([[y[groups[0]]][0],[y[groups[1]]][0]],labels)
                if gini < best_score:
                    best_score = gini
                    best_idx = i,j
                    best_groups = groups

        return best_idx,best_score,best_groups
    
    def _create_leaf(self,y):
        return Leaf(decision = np.bincount(y).argmax())
    
    def _bottom_up_pruning(self,X,y,tree_list):
        score = self.score(X,y)
        for new_tree in tree_list:
            tmp_decision_tree = deepcopy(self)
            tmp_decision_tree.tree = new_tree
            if tmp_decision_tree.score(X,y) >= (score - self.pruning_eps) :
                self.tree = new_tree # Pruning is successful
                self._bottom_up_pruning(X,y,[]) # Try pruning some more based on the new tree  
                break 



class DecisionTreeRegressor(BaseDecisionTree,BaseRegressor):
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

    def __init__(self,max_depth=10,min_samples_split=2,categorial_features = [],pruning_eps=2,metric = "mse"):
        super().__init__(max_depth,min_samples_split,categorial_features,pruning_eps)
        assert metric in ["mse","mae"], "Metric must one of the followwing : mse,mae"
        self.metric = metric # ignored for now


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
    
    def _create_leaf(self,y):
        return Leaf(decision = np.mean(y))
    
    def _bottom_up_pruning(self,X,y):
        tree_list = self.get_pruned_trees(X,y)
        score = self.score(X,y)
        print('new list')
        for new_tree in tree_list:
            tmp_decision_tree = deepcopy(self)
            tmp_decision_tree.tree = new_tree
            if tmp_decision_tree.score(X,y) <= (score):#+ self.pruning_eps) :
                self.tree = new_tree 
                self._bottom_up_pruning(X,y) # Try pruning some more based on the new tree  
                break 
