"""
 code for dealing with Bayesian composite models

For a model to be useable here, it should support the following API:

  - _ClassifyExample(example)_, returns a classification

Other compatibility notes:

 1) To use _Composite.Grow_ there must be some kind of builder
    functionality which returns a 2-tuple containing (model,percent accuracy).

 2) The models should be pickleable

 3) It would be very happy if the models support the __cmp__ method so that
    membership tests used to make sure models are unique work.



"""
from __future__ import annotations
import numpy as numpy
from rdkit.ML.Composite import Composite
import rdkit.ML.Composite.Composite
__all__ = ['BayesComposite', 'BayesCompositeToComposite', 'Composite', 'CompositeToBayesComposite', 'numpy']
class BayesComposite(rdkit.ML.Composite.Composite.Composite):
    """
    a composite model using Bayesian statistics in the Decision Proxy
    
    
        **Notes**
    
        - typical usage:
    
           1) grow the composite with AddModel until happy with it
    
           2) call AverageErrors to calculate the average error values
    
           3) call SortModels to put things in order by either error or count
    
           4) call Train to update the Bayesian stats.
    
      
    """
    def ClassifyExample(self, example, threshold = 0, verbose = 0, appendExample = 0):
        """
         classifies the given example using the entire composite
        
              **Arguments**
        
               - example: the data to be classified
        
               - threshold:  if this is a number greater than zero, then a
                  classification will only be returned if the confidence is
                  above _threshold_.  Anything lower is returned as -1.
        
              **Returns**
        
                a (result,confidence) tuple
        
            
        """
    def Train(self, data, verbose = 0):
        ...
    def __init__(self):
        ...
def BayesCompositeToComposite(obj):
    """
     converts a BayesComposite to a Composite.Composite
    
      
    """
def CompositeToBayesComposite(obj):
    """
     converts a Composite to a BayesComposite
    
       if _obj_ is already a BayesComposite or if it is not a _Composite.Composite_ ,
        nothing will be done.
    
      
    """
