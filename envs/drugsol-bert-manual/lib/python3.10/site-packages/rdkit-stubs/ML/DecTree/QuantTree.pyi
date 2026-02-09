"""
 Defines the class _QuantTreeNode_, used to represent decision trees with automatic
 quantization bounds

  _QuantTreeNode_ is derived from _DecTree.DecTreeNode_

"""
from __future__ import annotations
import rdkit.ML.DecTree.DecTree
from rdkit.ML.DecTree import DecTree
from rdkit.ML.DecTree import Tree
import typing
__all__ = ['DecTree', 'QuantTreeNode', 'Tree']
class QuantTreeNode(rdkit.ML.DecTree.DecTree.DecTreeNode):
    """
    
    
      
    """
    __hash__: typing.ClassVar[None] = None
    def ClassifyExample(self, example, appendExamples = 0):
        """
         Recursively classify an example by running it through the tree
        
              **Arguments**
        
                - example: the example to be classified
        
                - appendExamples: if this is nonzero then this node (and all children)
                  will store the example
        
              **Returns**
        
                the classification of _example_
        
              **NOTE:**
                In the interest of speed, I don't use accessor functions
                here.  So if you subclass DecTreeNode for your own trees, you'll
                have to either include ClassifyExample or avoid changing the names
                of the instance variables this needs.
        
            
        """
    def GetQuantBounds(self):
        ...
    def SetQuantBounds(self, qBounds):
        ...
    def __cmp__(self, other):
        ...
    def __eq__(self, other):
        ...
    def __init__(self, *args, **kwargs):
        ...
    def __lt__(self, other):
        ...
    def __str__(self):
        """
         returns a string representation of the tree
        
              **Note**
        
                this works recursively
        
            
        """
