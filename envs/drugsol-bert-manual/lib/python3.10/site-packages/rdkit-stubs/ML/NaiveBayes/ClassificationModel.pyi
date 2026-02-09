"""
 Defines Naive Baysean classification model
   Based on development in: Chapter 6 of "Machine Learning" by Tom Mitchell

"""
from __future__ import annotations
import numpy as numpy
from rdkit.ML.Data import Quantize
__all__ = ['NaiveBayesClassifier', 'Quantize', 'numpy']
class NaiveBayesClassifier:
    """
    
        _NaiveBayesClassifier_s can save the following pieces of internal state, accessible via
        standard setter/getter functions:
    
        1) _Examples_: a list of examples which have been predicted
    
        2) _TrainingExamples_: List of training examples - the descriptor value of these examples
          are quantized based on info gain using ML/Data/Quantize.py if necessary
    
        3) _TestExamples_: the list of examples used to test the model
    
        4) _BadExamples_ : list of examples that were incorrectly classified
    
        4) _QBoundVals_: Quant bound values for each varaible - a list of lists
    
        5) _QBounds_ : Number of bounds for each variable
    
        
    """
    def ClassifyExample(self, example, appendExamples = 0):
        """
         Classify an example by summing over the conditional probabilities
                The most likely class is the one with the largest probability
                
        """
    def ClassifyExamples(self, examples, appendExamples = 0):
        ...
    def GetBadExamples(self):
        ...
    def GetClassificationDetails(self):
        """
         returns the probability of the last prediction 
        """
    def GetExamples(self):
        ...
    def GetName(self):
        ...
    def GetTestExamples(self):
        ...
    def GetTrainingExamples(self):
        ...
    def NameModel(self, varNames):
        ...
    def SetBadExamples(self, examples):
        ...
    def SetExamples(self, examples):
        ...
    def SetName(self, name):
        ...
    def SetTestExamples(self, examples):
        ...
    def SetTrainingExamples(self, examples):
        ...
    def __init__(self, attrs, nPossibleVals, nQuantBounds, mEstimateVal = -1.0, useSigs = False):
        """
         Constructor
        
                
        """
    def _computeQuantBounds(self):
        ...
    def trainModel(self):
        """
         We will assume at this point that the training examples have been set
        
                We have to estmate the conditional probabilities for each of the (binned) descriptor
                component give a outcome (or class). Also the probabilities for each class is estimated
                
        """
def _getBinId(val, qBounds):
    ...
