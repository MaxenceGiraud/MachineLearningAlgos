import numpy as np

def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res
def onehot(y,i):
    return 1-np.array(list(bin(y[i]+1)[2:].zfill(2))).astype(int)

class InputUnit:
    def __init__(self,data):
        self.data = data #list of preceding neurons
        self.n = data.shape[0] #dataset size
        self.k = 0 #layer number
        self.z = 0 #unit output

    def plug(self,OutputUnit):
        '''
        @param OutputUnit : Unit we want to connect to
        Plug this unit to the OutputUnit
        '''
        OutputUnit.preceding.append(self) # add to output unit's preceding list the current unit
        
    def forward(self,i):
        '''
        @param i : index on which we want to perform a forward pass on 
        '''
        self.z = self.data[i] # ith data
        return self.z


class NeuralUnit:
    #Constructor
    def __init__(self,k,u):
        self.u = u #unit number
        self.preceding = [] #list of
        self.npr = 0 #length of list
        self.following = [] #list of
        self.nfo = 0 #length of list
        self.k = k #layer number
        self.w = 0 #unit weights
        self.b = 0 #unit intercept
        self.z = 0 #unit output
        
        self.delta = 0
        self.w_grad = 0 
        self.b_grad = 0

    def reset_params(self):
        '''
        Reset params of this unit
        '''
        self.npr = len(self.preceding)
        self.nfo = len(self.following)
        self.w = np.random.randn(len(self.preceding))
        self.b = np.random.randn()
        self.delta = np.zeros(self.w.shape)
        self.w_grad = np.zeros(self.w.shape)
        
    def plug(self,OutputUnit):
        '''
        @param OutputUnit : Unit we want to connect to
        Plug this unit to the OutputUnit
        '''
        self.following.append(OutputUnit)  # add to following list the output unit
        OutputUnit.preceding.append(self) # add to output unit's preceding list the current unit
    
    def forward(self,i):
        '''
        @param i : index on which we want to perform a forward pass on 
        '''
        x = np.zeros(self.npr)
        for j in range(self.npr): # for each preceding neuron
            x[j] = self.preceding[j].forward(i) # output preceding neuron
        self.z = sigmoid(self.w @ x + self.b) # computing unit output
        return self.z
    
    def backprop(self,i,deltas):
        '''
        @param i :index for which we want to compute the gradient
        @param delta : delta derivatives from the following layer
        '''
        for v in range(self.npr) :
            self.delta[v] = self.z*(1-self.z)* self.w[v] * deltas[self.u] #delta derivative
            self.w_grad[v] = self.z*(1-self.z) * self.preceding[v].z * deltas[self.u] # parameter derivate weights
        self.b_grad =self.z*(1-self.z) * deltas[self.u] # parameter derivate intercept

class Loss:
    #Constructor
    def __init__(self,y,k):
        self.preceding = [] #list of preceding neurons
        self.npr = len(self.preceding) #length of list preceding
        self.y = y #array of class labels of the training data
        self.k = k #layer index
        self.l = 0 # Loss
        self.delta = np.zeros((1,)) # derivative
        
    def forward(self,i):
        '''
        @param i : index on which we want to perform a forward pass on 
        '''
        zin = self.preceding[0].forward(i) # output of previous layer
        self.l = np.array([-np.log(1-zin),-np.log(zin)]) @ onehot(self.y,i) 
        return self.l
    
    def backprop(self,i):
        zin = self.preceding[0].forward(i) # output of previous layer
        self.delta = np.array([np.array([1/(1-zin),-1/zin]) @ onehot(self.y,i)])  # derivative

    
class MLP:
    #Constructor
    def __init__(self,X,y,archi):
        self.archi = archi
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.K = len(archi) # number of layers (including loss layer and input layer)

        #### CREATING NETWORK
        net = np.zeros((np.max(archi),len(archi)),dtype=object)
        ## INPUT UNITS
        for i in range (archi[0]):
            net[i,0]=(InputUnit(data= X[:,i]))
        ## NEURAL UNITS
        for layer_nb in range(len(archi[1:-1])): ## For each layer
            for unit_nb in range(archi[layer_nb+1]): # For each unit 
                net[unit_nb,layer_nb+1]=(NeuralUnit(k=layer_nb,u=unit_nb)) # Creating the neural units
                
                for preceding_unit_nb in range(archi[layer_nb]): # plug each precdeding units to the current one
                    net[preceding_unit_nb,layer_nb].plug(net[unit_nb,layer_nb+1])
                    
                net[unit_nb,layer_nb+1].reset_params() # Reset the params of the neural units
                
        ## LOSS 
        net[0,-1] = Loss(y=self.y,k=len(archi))
        net[0,-2].plug(net[0,-1])
        net[0,-1].npr = len(net[0,-1].preceding)
        self.net = net
        
    def forward(self,i):
        '''
        @param i : index on which we want to perform a forward pass on 
        '''
        return self.net[0,-1].forward(i) # Forward of the loss layer
    
    def backprop(self,i):
        self.net[0,-1].backprop(i) ## on the loss layer
        deltas = self.net[0,-1].delta # update delta
        
        for k in range(len(self.archi)-2,0,-1): # On the Neural units
            deltas_new = np.zeros((self.net[0,k].npr,))
            for u in range(self.archi[k]):
                self.net[u,k].backprop(i,deltas)    
                deltas_new += self.net[u,k].delta
            deltas = deltas_new
            
    def update(self,eta):
        '''
        @param eta : learning rate
        '''
        for k in range(len(self.archi)-2,0,-1): # For each neural layer
            for u in range(self.archi[k]): # For each neural unit
                self.net[u,k].w -= eta * self.net[u,k].w_grad
                self.net[u,k].b -= eta * self.net[u,k].b_grad
        
    def train(self, epochs,eta):
        '''
        @param epochs :number of epochs
        @param eta : learning rate
        Train the neural network
        '''
        for epoch in range(epochs):
            for i in range(self.n): # for each data point
                self.forward(i)
                self.backprop(i)
                self.update(eta)
                
    def predict(self,i):
        '''
        predict 1 data point
        '''
        return self.net[0,-2].forward(i)