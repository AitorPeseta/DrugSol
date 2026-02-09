"""
 Define the class _KNNRegressionModel_, used to represent a k-nearest neighbhors
regression model

    Inherits from _KNNModel_
"""
from __future__ import annotations
import rdkit.ML.KNN.KNNModel
from rdkit.ML.KNN import KNNModel
__all__ = ['KNNModel', 'KNNRegressionModel']
class KNNRegressionModel(rdkit.ML.KNN.KNNModel.KNNModel):
    """
     This is used to represent a k-nearest neighbor classifier
    
      
    """
    def GetBadExamples(self):
        ...
    def NameModel(self, varNames):
        ...
    def PredictExample(self, example, appendExamples = 0, weightedAverage = 0, neighborList = None):
        """
         Generates a prediction for an example by looking at its closest neighbors
        
            **Arguments**
        
              - examples: the example to be classified
        
              - appendExamples: if this is nonzero then the example will be stored on this model
        
              - weightedAverage: if provided, the neighbors' contributions to the value will be
                                 weighed by their reciprocal square distance
        
              - neighborList: if provided, will be used to return the list of neighbors
        
            **Returns**
        
              - the classification of _example_
        
            
        """
    def SetBadExamples(self, examples):
        ...
    def __init__(self, k, attrs, dfunc, radius = None):
        ...
    def type(self):
        ...
