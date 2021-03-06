from .base_kernel import KernelConcat

class KernelFusion(KernelConcat):
    ''' Kernel that combine multiple kernels on different features of the dimensions '''
    def __init__(self,kernels,features,normalize=True):
        assert len(kernels) == len(features), "The size of the feature list and the kernel list does not match"

        self.kernels = kernels
        self.features = features
        self.normalize = normalize

        self.operation = lambda a,b: a+b
        
        if self.normalize : 
            for i in range(len(self.kernels)):
                self.kernels[i] = self.kernels[i].normalize()

        self.to_precompute = set()
        for k in self.kernels :
            self.to_precompute =self.to_precompute.union(k.to_precompute)
        self.precomputed = {}

    def _compute_kernel(self,x,y,**kwargs):
        self.precomputed = self.get_precomputed(x,y,**kwargs)

        out = 0
        for i in range(len(self.kernels)):
                out = out + self.kernels[i](x[:,self.features[i]],y[:,self.features[i]],**self.precomputed)
        return out