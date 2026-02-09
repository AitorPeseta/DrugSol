"""
 ID3 Decision Trees

  contains an implementation of the ID3 decision tree algorithm
  as described in Tom Mitchell's book "Machine Learning"

  It relies upon the _Tree.TreeNode_ data structure (or something
    with the same API) defined locally to represent the trees

"""
from __future__ import annotations
import numpy as numpy
from rdkit.ML.DecTree import DecTree
from rdkit.ML.InfoTheory import entropy
__all__ = ['CalcTotalEntropy', 'DecTree', 'GenVarTable', 'ID3', 'ID3Boot', 'entropy', 'numpy']
def CalcTotalEntropy(examples, nPossibleVals):
    """
     Calculates the total entropy of the data set (w.r.t. the results)
    
       **Arguments**
    
        - examples: a list (nInstances long) of lists of variable values + instance
                  values
        - nPossibleVals: a list (nVars long) of the number of possible values each variable
          can adopt.
    
       **Returns**
    
         a float containing the informational entropy of the data set.
    
      
    """
def GenVarTable(examples, nPossibleVals, vars):
    """
    Generates a list of variable tables for the examples passed in.
    
        The table for a given variable records the number of times each possible value
        of that variable appears for each possible result of the function.
    
      **Arguments**
    
        - examples: a list (nInstances long) of lists of variable values + instance
                  values
    
        - nPossibleVals: a list containing the number of possible values of
                       each variable + the number of values of the function.
    
        - vars:  a list of the variables to include in the var table
    
    
      **Returns**
    
          a list of variable result tables. Each table is a Numeric array
            which is varValues x nResults
      
    """
def ID3(examples, target, attrs, nPossibleVals, depth = 0, maxDepth = -1, **kwargs):
    """
     Implements the ID3 algorithm for constructing decision trees.
    
        From Mitchell's book, page 56
    
        This is *slightly* modified from Mitchell's book because it supports
          multivalued (non-binary) results.
    
        **Arguments**
    
          - examples: a list (nInstances long) of lists of variable values + instance
                  values
    
          - target: an int
    
          - attrs: a list of ints indicating which variables can be used in the tree
    
          - nPossibleVals: a list containing the number of possible values of
                       every variable.
    
          - depth: (optional) the current depth in the tree
    
          - maxDepth: (optional) the maximum depth to which the tree
                       will be grown
    
        **Returns**
    
         a DecTree.DecTreeNode with the decision tree
    
        **NOTE:** This code cannot bootstrap (start from nothing...)
              use _ID3Boot_ (below) for that.
      
    """
def ID3Boot(examples, attrs, nPossibleVals, initialVar = None, depth = 0, maxDepth = -1, **kwargs):
    """
     Bootstrapping code for the ID3 algorithm
    
        see ID3 for descriptions of the arguments
    
        If _initialVar_ is not set, the algorithm will automatically
         choose the first variable in the tree (the standard greedy
         approach).  Otherwise, _initialVar_ will be used as the first
         split.
    
      
    """
