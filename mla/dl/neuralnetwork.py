import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from .optimizer.gradientdescent import GradientDescent
from .layers.inputlayer import InputLayer
from .layers.loss import Loss,MSE,MAE
from .layers.dense import Dense

def get_colorshape(arg):
    ''' Given a layer class, outputs a color and a shape for each layer type'''
    switcher = { 
        'InputLayer': ('black','o'), 
        'Dense' : ('blue','o'),
        'Conv1D' : ('lightcoral','s'),
        'Conv2D' : ('red','s'),
        'Conv3D' : ('maroon','s'),
        'Recurrent' : ('limegreen','$\circlearrowleft$'),
        'LSTM' : ('forestgreen','$\circlearrowleft$'),
        'GRU' : ('darkgreen','$\circlearrowleft$'),
        'Transformer' : ('mediumseagreen','$\circlearrowleft$'),
        'Flatten' : ('dimgrey','$\|$'),
        'Reshape' : ('dimgrey','$\|$'),
    } 
    if arg not in switcher :
        raise Exception("Layer type not yep supported for display")
    return switcher.get(arg,None) 

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique),bbox_to_anchor=(0,0.75))

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

    def get_layers_to_update(self):
        return self.layers[1:]

    def update(self,lr,noise_std=0):
        for layer in self.layers[1:]:
            layer.update(lr,noise_std=noise_std)

    def fit(self,X,y,optimizer=GradientDescent()):
        optimizer.minimize(self,X,y)
    
    def predict(self,X):
        return self.output_layer.forward(X)

    def score(self,X,y):
        y_hat = self.predict(X)
        return self.loss.loss_function(y,y_hat)

    def get_list_layers_todisplay(self):
        return self.layers

    def display(self,print_connections=True):
        layers = self.get_list_layers_todisplay()
        fig, ax = plt.subplots()
        nu = 0
        for i in range(len(layers)) : 
            layer_type = layers[i].__class__.__name__
            color,mark = get_colorshape(layer_type)
            if layer_type in ['Flatten','Reshape'] : # Layer not to display (e.g. flatten)
                ax.scatter(i,0,marker=mark,c=color,s=10000)
                continue
            old_nu = nu
            nu = layers[i].units

            ax.scatter(np.ones(nu)*i,np.linspace(int(-nu/2),int(nu/2),nu),s=350,zorder=1,c=color,marker=mark,label=layer_type)

            # plot lines/ connection
            if i!=0 and print_connections :
                for j in np.linspace(int(-old_nu/2),int(old_nu/2),old_nu):
                    for k in np.linspace(int(-nu/2),int(nu/2),nu):
                        ax.plot([last_displayed,i],[j,k],c='gray',zorder=-1)
            last_displayed = i
                
        legend_without_duplicate_labels(ax)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def summary(self):
        cols = ['Layer Type','Output Shape','Trainable Parameters']
        layers  = self.get_list_layers_todisplay()
        summary = []
        params_total = 0
        for l in layers : 
            summary.append([l.__class__.__name__,l.output_shape,l.nparams])
            params_total += l.nparams

        
        summary = pd.DataFrame(summary,columns=cols)
        print(summary.to_markdown(index=False))

        print("\nTotal trainable parameters :", params_total)