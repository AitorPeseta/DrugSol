"""
 Define the class _KNNModel_, used to represent a k-nearest neighbhors model

"""
from __future__ import annotations
from rdkit.DataStructs.TopNContainer import TopNContainer
__all__ = ['KNNModel', 'TopNContainer']
class KNNModel:
    """
     This is a base class used by KNNClassificationModel
      and KNNRegressionModel to represent a k-nearest neighbor predictor. In general
      one of this child classes needs to be instantiated.
    
      _KNNModel_s can save the following pieces of internal state, accessible via
        standard setter/getter functions - the child object store additional stuff:
    
        1) _Examples_: a list of examples which have been predicted (either classified
                        or values predicted)
    
        2) _TrainingExamples_: List of training examples (since this is a KNN model these examples
                               along with the value _k_ below define the model)
    
        3) _TestExamples_: the list of examples used to test the model
    
        4) _k_: the number of closest neighbors used for prediction
    
      
    """
    def GetExamples(self):
        ...
    def GetName(self):
        ...
    def GetNeighbors(self, example):
        """
         Returns the k nearest neighbors of the example
        
            
        """
    def GetTestExamples(self):
        ...
    def GetTrainingExamples(self):
        ...
    def SetExamples(self, examples):
        ...
    def SetName(self, name):
        ...
    def SetTestExamples(self, examples):
        ...
    def SetTrainingExamples(self, examples):
        ...
    def __init__(self, k, attrs, dfunc, radius = None):
        ...
    def _setup(self, k, attrs, dfunc, radius):
        ...
