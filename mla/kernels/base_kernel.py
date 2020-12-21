from abc import abstractmethod

class BaseKernel:
    def __init__(self):
        pass

    @abstractmethod
    def f(self):
        pass