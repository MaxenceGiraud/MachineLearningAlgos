class BaseLayer:
    def __init__(self):
        pass

    @property
    def nparams(self):
        return 0 
    
    def plug(self,inputlayer):
        self.input_shape = inputlayer.output_shape[0]

        self.input_unit = inputlayer
        inputlayer.output_unit = self

        self.zin = 0
    
    def forward(self,X,*args,**kwargs):
        self.zin = self.input_unit.forward(X) 
        self.zout = self.zin
        return self.zout
    
    def backprop(self,delta,*args,**kwargs):
        return delta

    def get_gradients(self,*args,**kwargs):
        return ()
    
    def update_weights(self,*args,**kwargs):
        pass