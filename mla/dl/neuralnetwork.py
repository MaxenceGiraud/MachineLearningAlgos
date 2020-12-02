import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations 
from .optimizer.gradientdescent import GradientDescent
from .layers.inputlayer import InputLayer
from .layers.loss import Loss,MSE,MAE
from .layers.dense import Dense

def get_color(arg):
    ''' Given a layer class, outputs a color for each layer type'''
    switcher = { 
        'InputLayer': 'black', 
        'Dense' : 'blue',
        'Convolution' : 'red',
        'Recurrent' : 'green',
        'Transformer' : 'yellow',
    } 
    if arg not in switcher :
        raise Exception("Layer type not yep supported for display")
    return switcher.get(arg,None) 

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

class NeuralNetwork:
    '''Neural Network  '''
    def __init__(self,input_shape,loss=MAE()):
        self.layers = [InputLayer(input_shape)]
        self.loss = loss
    
    @property
    def output_layer(self):
        return self.layers[-1]

    def add(self,layer):
        layer.plug(self.output_layer)
        self.layers.append(layer)
        self.loss.plug(self.output_layer)

    def forward(self,X,y):
        return self.loss.forward(X,y)

    def backprop(self,y):
        delta = self.loss.backprop(y)
        delta_loss = np.copy(delta)
        for i in range(len(self.layers)-1,0,-1):
            delta = self.layers[i].backprop(delta)
        
        return delta_loss

    
    def update(self,lr,noise_std=0):
        for layer in self.layers[1:]:
            layer.update(lr,noise_std)

    def fit(self,X,y,optimizer=GradientDescent()):
        optimizer.minimize(self,X,y)
    
    def predict(self,X):
        return self.output_layer.forward(X)

    def score(self,X,y):
        y_hat = self.predict(X)
        return self.loss.loss_function(y,y_hat)

    def get_list_layers_todisplay(self):
        return self.layers

    def display(self):
        layers = self.get_list_layers_todisplay()

        fig, ax = plt.subplots()

        # Input layers
        layer_type = layers[0].__class__.__name__
        color = get_color(layer_type)
        ax.scatter(0,0,s=350,c=color,label=layer_type)
        nu = 1
        for i in range(1,len(layers)) : 
            layer_type = layers[i].__class__.__name__
            color = get_color(layer_type)
            old_nu = nu
            nu = layers[i].units
            ax.scatter(np.ones(nu)*i,np.linspace(int(-nu/2),int(nu/2),nu),s=350,zorder=1,c=color,label=layer_type)
            # plot connexion
            for j in np.linspace(int(-old_nu/2),int(old_nu/2),old_nu):
                for k in np.linspace(int(-nu/2),int(nu/2),nu):
                    ax.plot([i-1,i],[j,k],c='gray',zorder=-1)
                
        legend_without_duplicate_labels(ax)
        plt.axis('off')
        plt.show()