import numpy as np
from ..base import BaseClassifier,BaseRegressor
from copy import deepcopy
import matplotlib.pyplot as plt
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
    
    @property
    def width(self):
        raise NotImplementedError
    
    @property
    def size(self):
        '''return number of leaves and nodes in the tree '''
        return self.left.size + self.right.size + 1

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
    
    def _count_preleaf(self):
        if isinstance(self.left,Leaf) : 
            if isinstance(self.right,Leaf) :
                return 1
            else :
                return self.right._count_preleaf()
        elif isinstance(self.right,Leaf) :
            return self.left._count_preleaf()
        else : 
            return self.right._count_preleaf() + self.left._count_preleaf()
            
    
    def _define_position(self,position = (0,0)):
        self.pos = position
        d = self.depth 
        self.left._define_position((self.pos[0]-2**d,self.pos[1]-1))
        self.right._define_position((self.pos[0]+2**d,self.pos[1]-1))

    def _get_tree_data(self):
        d = {}
        d['type'] = 'node'
        d['pos'] = self.pos
        d['feature'] = self.idx_feature
        d['criterion'] = self.criterion
        d['categorical'] = self.categorial

        arr = [d]
        l = self.left._get_tree_data()
        r = self.right._get_tree_data()
        l[0]['parent_pos'] = self.pos
        r[0]['parent_pos'] = self.pos
        arr.extend(l)
        arr.extend(r)

        return arr

    def display(self,pos=(0,0)):
        self._define_position(pos)
        tree = self._get_tree_data()

        for n in tree :
            pos = np.array(n['pos'])
            if n['type'] == 'node' :
                if n['categorical'] :
                    sign = '='
                else : 
                    sign = '<'
                
                if isinstance(n['criterion'],float):
                    n['criterion'] = int(n['criterion']*100)/100
                s= 'Feat '+str(n['feature'])+sign+str(n['criterion'])
            elif n['type'] == 'leaf' :
                if isinstance(n['decision'],float):
                    n['decision'] = int(n['decision']*100)/100
                s = str(n['decision'])
                
            else : 
                raise Exception("Node type cannot be displayed")
            plt.text(*pos,s,bbox=dict(facecolor='white'),horizontalalignment='center', verticalalignment='center',)
            if 'parent_pos' in n : ## If not top of the tree
                line = np.array([n['pos'],n['parent_pos']])
                plt.plot(line[:,0],line[:,1])
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        
   
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
    
    @property
    def size(self):
        return 1
    
    def split_data(self,X):
        pass

    def predict(self,X):
        return np.ones(X.shape[0])*self.decision
    
    def _define_position(self,position):
        self.pos = position
    
    def _get_tree_data(self):
        d = {}
        d['type'] = 'leaf'
        d['pos'] = self.pos
        d['feature'] = self.idx_feature
        d['criterion'] = self.criterion
        d['decision'] = self.decision

        return [d]


    
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
    def __init__(self,max_depth=6,min_samples_split=2,categorial_features = [],eps=0.02,pruning_eps=0.05):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.categorial_features  = categorial_features
        self.eps = eps
        self.pruning_eps = pruning_eps
        self.tree = None
    
    @property
    def depth(self):
        return self.tree.depth
    
    def display(self):
        if self.tree is None :
            raise Exception('The tree is not created')
        else :
            self.tree.display()
    
    def fit(self,X,y,prune = True):
        self.tree = self.build_node(X,y,0)  
        if prune : 
            self.prune(X,y)

    def _create_leaf(self,y):
        raise Exception

    def get_best_split(self,X,y):
        raise Exception
    
    def _create_pruned_tree(self,node,X,y,n):
        if isinstance(node.left,Leaf)  :
            if isinstance(node.right,Leaf) :
                return True,n
            else : 
                left,right = node.split_data(X)
                temp_right,n = self._create_pruned_tree(node.right,X[right],y[right],n)
                if temp_right == 'done' :
                    return 'done',n
                elif temp_right == True :
                    n -= 1
                    if n == 0 :
                        node.right = self._create_leaf(y)
                        return 'done',n

        elif isinstance(node.right,Leaf) :
            left,right = node.split_data(X)
            temp_left,n = self._create_pruned_tree(node.left,X[left],y[left],n)
            
            if temp_left == 'done' :
                return 'done',n
            elif temp_left == True :
                n -= 1
                if n == 0 :
                    node.left = self._create_leaf(y)
                    return 'done',n

        elif not isinstance(node,Leaf): 
            left,right = node.split_data(X)
            temp_left,n = self._create_pruned_tree(node.left,X[left],y[left],n)
            
            if temp_left == 'done' :
                return 'done',n
            elif temp_left == True :
                n -= 1
                if n == 0 :
                    node.left = self._create_leaf(y)
                    return 'done',n

            temp_right,n = self._create_pruned_tree(node.right,X[right],y[right],n)
            if temp_right == 'done' :
                return 'done',n
            elif temp_right == True :
                n -= 1
                if n == 0 :
                    node.right = self._create_leaf(y)
                    return 'done',n
        return False,n
                
    def get_pruned_trees(self,X,y):
        tree_list = []
        if self.tree.depth > 2 : 
            # Get all possible pruned trees  
            n_pruned_trees = self.tree._count_preleaf()
            for i in range(n_pruned_trees):
                new_tree = deepcopy(self.tree)
                self._create_pruned_tree(new_tree,X,y,n=i+1)
                tree_list.append(new_tree)
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
        idx,score,groups_idx = self.get_best_split(X,y) # Get the best split
        
        current_node = Node(X[idx[0],idx[1]],idx[1],score,categorial= (idx[1] in self.categorial_features))
        # Create a leaf (end of the tree)
        if self.max_depth <= depth+1 or self.min_samples_split >= X.shape[0] or score < self.eps:
            current_node = self._create_leaf(y)
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
                # Split data
                if j in self.categorial_features :
                    groups = np.where(X[:,j]==X[i,j])[0], np.where(X[:,j]!=X[i,j])[0]
                else :
                    groups = np.where(X[:,j]<X[i,j])[0], np.where(X[:,j]>=X[i,j])[0]
                gini = gini_index([[y[groups[0]]][0],[y[groups[1]]][0]],labels)
                
                if gini < best_score: # If better keep
                    best_score = gini
                    best_idx = i,j
                    best_groups = groups

        return best_idx,best_score,best_groups
    
    def _create_leaf(self,y):
        return Leaf(decision = np.bincount(y).argmax())
    
    def _bottom_up_pruning(self,X,y):
        tree_list = self.get_pruned_trees(X,y)
        score = self.score(X,y)
        for new_tree in tree_list:
            tmp_decision_tree = deepcopy(self)
            tmp_decision_tree.tree = new_tree
            if tmp_decision_tree.score(X,y) >= (score - self.pruning_eps) : # Pruning is successful
                self.tree = new_tree 
                self._bottom_up_pruning(X,y) # Try pruning some more based on the new tree  
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

    def __init__(self,max_depth=10,min_samples_split=2,categorial_features = [],eps=0,pruning_eps=2,metric = "mse"):
        super().__init__(max_depth=max_depth,min_samples_split=min_samples_split,categorial_features=categorial_features,eps=eps,pruning_eps=pruning_eps)
        assert metric in ["mse","mae"], "Metric must one of the followwing : mse,mae"
        self.metric = metric # ignored for now


    def get_best_split(self,X,y):
        ''' get the best one split possible '''
        best_score = np.inf
        labels  = np.unique(y)
        for j in range(X.shape[1]):
            # iterate on unique value of features
            for i in np.unique(X[:,j],return_index=True)[1]:
                # Split data
                if j in self.categorial_features :
                    groups = np.where(X[:,j]==X[i,j])[0], np.where(X[:,j]!=X[i,j])[0]
                else :
                    groups = np.where(X[:,j]<X[i,j])[0], np.where(X[:,j]>=X[i,j])[0]
                
                n_samples = np.sum([len(group) for group in groups])
                score = np.sum([np.square(group-np.mean(group)).sum() * len(group)/n_samples for group in groups])   
                if score < best_score: # If better keep
                    best_score = score
                    best_idx = i,j
                    best_groups = groups

        return best_idx,best_score,best_groups
    
    def _create_leaf(self,y):
        return Leaf(decision = np.mean(y))
    
    def _bottom_up_pruning(self,X,y):
        tree_list = self.get_pruned_trees(X,y)
        score = self.score(X,y)
        for new_tree in tree_list:
            tmp_decision_tree = deepcopy(self)
            tmp_decision_tree.tree = new_tree
            if tmp_decision_tree.score(X,y) <= (score +self.pruning_eps) : # Successful pruning
                self.tree = new_tree 
                self._bottom_up_pruning(X,y) # Try pruning some more based on the new tree  
                break 
