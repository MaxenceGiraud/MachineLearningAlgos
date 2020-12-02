class BaseLayer:
    def __init__(self):
        pass

    @property
    def nparams(self):
        return 0 
    
    def plug(self,intputlayer):
        self.input_shape = intputlayer.output_shape[0]

        self.input_unit = intputlayer
        intputlayer.output_unit = self

        self.zin = 0