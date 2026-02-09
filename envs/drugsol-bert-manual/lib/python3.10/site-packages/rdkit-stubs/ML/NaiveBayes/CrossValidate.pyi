"""
 handles doing cross validation with naive bayes models
and evaluation of individual models

"""
from __future__ import annotations
from rdkit.ML.Data import SplitData
from rdkit.ML.NaiveBayes.ClassificationModel import NaiveBayesClassifier
__all__ = ['CMIM', 'CrossValidate', 'CrossValidationDriver', 'NaiveBayesClassifier', 'SplitData', 'makeNBClassificationModel']
def CrossValidate(NBmodel, testExamples, appendExamples = 0):
    ...
def CrossValidationDriver(examples, attrs, nPossibleValues, nQuantBounds, mEstimateVal = 0.0, holdOutFrac = 0.3, modelBuilder = makeNBClassificationModel, silent = 0, calcTotalError = 0, **kwargs):
    ...
def makeNBClassificationModel(trainExamples, attrs, nPossibleValues, nQuantBounds, mEstimateVal = -1.0, useSigs = False, ensemble = None, useCMIM = 0, **kwargs):
    ...
CMIM = None
