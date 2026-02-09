"""
 handles doing cross validation with k-nearest neighbors model

and evaluation of individual models

"""
from __future__ import annotations
from rdkit.ML.Data import SplitData
import rdkit.ML.KNN.DistFunctions
from rdkit.ML.KNN import DistFunctions
from rdkit.ML.KNN.KNNClassificationModel import KNNClassificationModel
from rdkit.ML.KNN.KNNRegressionModel import KNNRegressionModel
__all__ = ['CrossValidate', 'CrossValidationDriver', 'DistFunctions', 'KNNClassificationModel', 'KNNRegressionModel', 'SplitData', 'makeClassificationModel', 'makeRegressionModel']
def CrossValidate(knnMod, testExamples, appendExamples = 0):
    """
    
      Determines the classification error for the testExamples
    
      **Arguments**
    
        - tree: a decision tree (or anything supporting a _ClassifyExample()_ method)
    
        - testExamples: a list of examples to be used for testing
    
        - appendExamples: a toggle which is passed along to the tree as it does
          the classification. The trees can use this to store the examples they
          classify locally.
    
      **Returns**
    
        a 2-tuple consisting of:
          
    """
def CrossValidationDriver(examples, attrs, nPossibleValues, numNeigh, modelBuilder = makeClassificationModel, distFunc = rdkit.ML.KNN.DistFunctions.EuclideanDist, holdOutFrac = 0.3, silent = 0, calcTotalError = 0, **kwargs):
    """
     Driver function for building a KNN model of a specified type
    
      **Arguments**
    
        - examples: the full set of examples
    
        - numNeigh: number of neighbors for the KNN model (basically k in k-NN)
    
        - knnModel: the type of KNN model (a classification vs regression model)
    
        - holdOutFrac: the fraction of the data which should be reserved for the hold-out set
          (used to calculate error)
    
        - silent: a toggle used to control how much visual noise this makes as it goes
    
        - calcTotalError: a toggle used to indicate whether the classification error
          of the tree should be calculated using the entire data set (when true) or just
          the training hold out set (when false)
          
    """
def makeClassificationModel(numNeigh, attrs, distFunc):
    ...
def makeRegressionModel(numNeigh, attrs, distFunc):
    ...
