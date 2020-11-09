class InputLayer:
    def __init__(self,input_shape):
        self.input_shape  = input_shape
        self.output_shape  = self.input_shape

        self.intput_unit = None
        self.output_unit = None

    def forward(self,X):
        return X