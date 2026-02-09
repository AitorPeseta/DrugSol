"""
 Contains functionality for doing tree pruning

"""
from __future__ import annotations
import copy as copy
import numpy as numpy
from rdkit.ML.DecTree import CrossValidate
from rdkit.ML.DecTree import DecTree
__all__ = ['CrossValidate', 'DecTree', 'MaxCount', 'PruneTree', 'copy', 'numpy']
def MaxCount(examples):
    """
     given a set of examples, returns the most common result code
    
         **Arguments**
    
            examples: a list of examples to be counted
    
         **Returns**
    
           the most common result code
    
        
    """
def PruneTree(tree, trainExamples, testExamples, minimizeTestErrorOnly = 1):
    """
     implements a reduced-error pruning of decision trees
    
         This algorithm is described on page 69 of Mitchell's book.
    
         Pruning can be done using just the set of testExamples (the validation set)
         or both the testExamples and the trainExamples by setting minimizeTestErrorOnly
         to 0.
    
         **Arguments**
    
           - tree: the initial tree to be pruned
    
           - trainExamples: the examples used to train the tree
    
           - testExamples: the examples held out for testing the tree
    
           - minimizeTestErrorOnly: if this toggle is zero, all examples (i.e.
             _trainExamples_ + _testExamples_ will be used to evaluate the error.
    
         **Returns**
    
           a 2-tuple containing:
    
              1) the best tree
    
              2) the best error (the one which corresponds to that tree)
    
        
    """
def _GetLocalError(node):
    ...
def _Pruner(node, level = 0):
    """
    Recursively finds and removes the nodes whose removals improve classification
    
           **Arguments**
    
             - node: the tree to be pruned.  The pruning data should already be contained
               within node (i.e. node.GetExamples() should return the pruning data)
    
             - level: (optional) the level of recursion, used only in _verbose printing
    
    
           **Returns**
    
              the pruned version of node
    
    
           **Notes**
    
            - This uses a greedy algorithm which basically does a DFS traversal of the tree,
              removing nodes whenever possible.
    
            - If removing a node does not affect the accuracy, it *will be* removed.  We
              favor smaller trees.
    
        
    """
def _testChain():
    ...
def _testRandom():
    ...
def _testSpecific():
    ...
_verbose: int = 0
