from ..base import BaseAny
from abc import abstractmethod

class BaseMAB(BaseAny):
    ''' Abstract Class for MAB algorithm'''
    
    @abstractmethod
    def chooseArmToPlay(self):
        pass

    @abstractmethod
    def receiveReward(self,arm,reward):
        pass

