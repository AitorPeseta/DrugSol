"""
 Defines the class SigTreeNode, used to represent trees that
 use signatures (bit vectors) to represent data.  As inputs (examples),
 SigTreeNode's expect 3-sequences: (label,sig,act)

  _SigTreeNode_ is derived from _DecTree.DecTreeNode_

"""
from __future__ import annotations
import copy as copy
from rdkit.DataStructs.VectCollection import VectCollection
from rdkit.ML.DecTree import DecTree
import rdkit.ML.DecTree.DecTree
__all__ = ['DecTree', 'SigTreeNode', 'VectCollection', 'copy']
class SigTreeNode(rdkit.ML.DecTree.DecTree.DecTreeNode):
    """
    
    
      
    """
    def ClassifyExample(self, example, appendExamples = 0):
        """
         Recursively classify an example by running it through the tree
        
              **Arguments**
        
                - example: the example to be classified, a sequence at least
                  2 long:
                   ( id, sig )
                  where sig is a BitVector (or something supporting __getitem__)
                  additional fields will be ignored.
        
                - appendExamples: if this is nonzero then this node (and all children)
                  will store the example
        
              **Returns**
        
                the classification of _example_
        
            
        """
    def NameModel(self, *args, **kwargs):
        ...
    def __init__(self, *args, **kwargs):
        ...
