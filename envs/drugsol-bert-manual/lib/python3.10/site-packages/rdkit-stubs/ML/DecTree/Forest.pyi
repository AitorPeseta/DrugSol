"""
 code for dealing with forests (collections) of decision trees

**NOTE** This code should be obsolete now that ML.Composite.Composite is up and running.

"""
from __future__ import annotations
import numpy as numpy
import pickle as pickle
from rdkit.ML.DecTree import CrossValidate
from rdkit.ML.DecTree import PruneTree
__all__ = ['CrossValidate', 'Forest', 'PruneTree', 'numpy', 'pickle']
class Forest:
    """
    a forest of unique decision trees.
    
        adding an existing tree just results in its count field being incremented
            and the errors being averaged.
    
        typical usage:
    
          1) grow the forest with AddTree until happy with it
    
          2) call AverageErrors to calculate the average error values
    
          3) call SortTrees to put things in order by either error or count
    
      
    """
    def AddTree(self, tree, error):
        """
         Adds a tree to the forest
        
            If an identical tree is already present, its count is incremented
        
            **Arguments**
        
              - tree: the new tree
        
              - error: its error value
        
            **NOTE:** the errList is run as an accumulator,
                you probably want to call AverageErrors after finishing the forest
        
            
        """
    def AverageErrors(self):
        """
         convert summed error to average error
        
              This does the conversion in place
            
        """
    def ClassifyExample(self, example):
        """
         classifies the given example using the entire forest
        
              **returns** a result and a measure of confidence in it.
        
              **FIX:** statistics sucks... I'm not seeing an obvious way to get
                   the confidence intervals.  For that matter, I'm not seeing
                   an unobvious way.
        
                   For now, this is just treated as a voting problem with the confidence
                   measure being the percent of trees which voted for the winning result.
            
        """
    def CollectVotes(self, example):
        """
         collects votes across every member of the forest for the given example
        
              **Returns**
        
                a list of the results
        
            
        """
    def GetAllData(self):
        """
         Returns everything we know
        
            **Returns**
        
              a 3-tuple consisting of:
        
                1) our list of trees
        
                2) our list of tree counts
        
                3) our list of tree errors
        
            
        """
    def GetCount(self, i):
        ...
    def GetDataTuple(self, i):
        """
         returns all relevant data about a particular tree in the forest
        
              **Arguments**
        
                i: an integer indicating which tree should be returned
        
              **Returns**
        
                a 3-tuple consisting of:
        
                  1) the tree
        
                  2) its count
        
                  3) its error
            
        """
    def GetError(self, i):
        ...
    def GetTree(self, i):
        ...
    def GetVoteDetails(self):
        """
         Returns the details of the last vote the forest conducted
        
              this will be an empty list if no voting has yet been done
        
            
        """
    def Grow(self, examples, attrs, nPossibleVals, nTries = 10, pruneIt = 0, lessGreedy = 0):
        """
         Grows the forest by adding trees
        
             **Arguments**
        
              - examples: the examples to be used for training
        
              - attrs: a list of the attributes to be used in training
        
              - nPossibleVals: a list with the number of possible values each variable
                (as well as the result) can take on
        
              - nTries: the number of new trees to add
        
              - pruneIt: a toggle for whether or not the tree should be pruned
        
              - lessGreedy: toggles the use of a less greedy construction algorithm where
                each possible tree root is used.  The best tree from each step is actually
                added to the forest.
        
            
        """
    def MakeHistogram(self):
        """
         creates a histogram of error/count pairs
        
            
        """
    def Pickle(self, fileName = 'foo.pkl'):
        """
         Writes this forest off to a file so that it can be easily loaded later
        
             **Arguments**
        
               fileName is the name of the file to be written
        
            
        """
    def SetCount(self, i, val):
        ...
    def SetDataTuple(self, i, tup):
        """
         sets all relevant data for a particular tree in the forest
        
              **Arguments**
        
                - i: an integer indicating which tree should be returned
        
                - tup: a 3-tuple consisting of:
        
                  1) the tree
        
                  2) its count
        
                  3) its error
            
        """
    def SetError(self, i, val):
        ...
    def SetTree(self, i, val):
        ...
    def SortTrees(self, sortOnError = 1):
        """
         sorts the list of trees
        
              **Arguments**
        
                sortOnError: toggles sorting on the trees' errors rather than their counts
        
            
        """
    def __getitem__(self, which):
        """
         allows forest[i] to work.  return the data tuple
        
            
        """
    def __init__(self):
        ...
    def __len__(self):
        """
         allows len(forest) to work
        
            
        """
    def __str__(self):
        """
         allows the forest to show itself as a string
        
            
        """
