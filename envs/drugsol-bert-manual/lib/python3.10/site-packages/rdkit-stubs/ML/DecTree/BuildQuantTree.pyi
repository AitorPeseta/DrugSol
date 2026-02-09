"""


"""
from __future__ import annotations
import numpy as numpy
from rdkit.ML.Data import Quantize
from rdkit.ML.DecTree import ID3
from rdkit.ML.DecTree import QuantTree
from rdkit.ML.InfoTheory import entropy
from rdkit import RDRandom as random
__all__ = ['BuildQuantTree', 'FindBest', 'ID3', 'QuantTree', 'QuantTreeBoot', 'Quantize', 'TestQuantTree', 'TestQuantTree2', 'TestTree', 'entropy', 'numpy', 'random']
def BuildQuantTree(examples, target, attrs, nPossibleVals, nBoundsPerVar, depth = 0, maxDepth = -1, exIndices = None, **kwargs):
    """
    
          **Arguments**
    
            - examples: a list of lists (nInstances x nVariables+1) of variable
              values + instance values
    
            - target: an int
    
            - attrs: a list of ints indicating which variables can be used in the tree
    
            - nPossibleVals: a list containing the number of possible values of
                         every variable.
    
            - nBoundsPerVar: the number of bounds to include for each variable
    
            - depth: (optional) the current depth in the tree
    
            - maxDepth: (optional) the maximum depth to which the tree
                         will be grown
          **Returns**
    
           a QuantTree.QuantTreeNode with the decision tree
    
          **NOTE:** This code cannot bootstrap (start from nothing...)
                use _QuantTreeBoot_ (below) for that.
        
    """
def FindBest(resCodes, examples, nBoundsPerVar, nPossibleRes, nPossibleVals, attrs, exIndices = None, **kwargs):
    ...
def QuantTreeBoot(examples, attrs, nPossibleVals, nBoundsPerVar, initialVar = None, maxDepth = -1, **kwargs):
    """
     Bootstrapping code for the QuantTree
    
          If _initialVar_ is not set, the algorithm will automatically
           choose the first variable in the tree (the standard greedy
           approach).  Otherwise, _initialVar_ will be used as the first
           split.
    
        
    """
def TestQuantTree():
    """
     Testing code for named trees
    
        The created pkl file is required by the unit test code.
        
    """
def TestQuantTree2():
    """
     testing code for named trees
    
        The created pkl file is required by the unit test code.
        
    """
def TestTree():
    """
     testing code for named trees
    
        
    """
