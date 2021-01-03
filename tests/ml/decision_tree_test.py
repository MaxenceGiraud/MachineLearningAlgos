import unittest
import numpy as np
import matplotlib.pyplot as plt
from mla.ml.decison_tree import Node,Leaf,BaseDecisionTree,DecisionTreeClassifier,DecisionTreeRegressor

class NodeTest(unittest.TestCase):
    def setUp(self):
        self.n = Node(0,1)
        self.n.left = Node(0,0)
        self.n.right = Leaf(0)

        self.n.left.right = Node(0,0)
        self.n.left.left = Leaf(0)

        self.n.left.right.left = Leaf(0)
        self.n.left.right.right = Leaf(0)

    def test_depth(self):
        self.assertEqual(self.n.depth,3)
    
    def test_size(self):
        self.assertEqual(self.n.size,7)

    def test_split(self):
        left,right = self.n.split_data(np.array([[0,0],[1,1],[2,-1]]))
        self.assertTrue(np.all(list(left)==[2]))
        self.assertTrue(np.all(list(right[0])==[0,1]))
    
    def test_countpreleaf(self):
        self.assertEqual(self.n._count_preleaf(),1)
    
    def test_display(self):
        plt.figure()
        self.n.display()
        self.assertEqual(plt.gcf().number,1)
        plt.close('all')

class DecisionTreeTest(unittest.TestCase):
    def setUp(self):
        self.d = DecisionTreeClassifier()

        self.n = Node(0.5,1)
        self.n.left = Node(0.25,1)
        self.n.right = Leaf(0)

        self.n.left.right = Node(0,0.5)
        self.n.left.left = Leaf(0)

        self.n.left.right.left = Leaf(0)
        self.n.left.right.right = Leaf(0)

        self.d.tree = self.n

    def test_get_pruned_trees(self):
        np.random.seed(46)
        tree_list = self.d.get_pruned_trees(np.random.random(size=100).reshape(50,2),np.random.randint(0,2,size=50))
        print(tree_list)

        self.assertEqual(len(tree_list),1)
        self.assertEqual(tree_list[0].depth,2)
        self.assertEqual(tree_list[0]._count_preleaf(),1)
        self.assertEqual(tree_list[0].size,5)

if __name__ == '__main__':
    unittest.main()