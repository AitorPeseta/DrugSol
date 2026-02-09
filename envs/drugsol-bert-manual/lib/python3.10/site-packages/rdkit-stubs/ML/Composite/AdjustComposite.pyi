"""
 functionality to allow adjusting composite model contents

"""
from __future__ import annotations
import copy as copy
import numpy as numpy
__all__ = ['BalanceComposite', 'copy', 'numpy']
def BalanceComposite(model, set1, set2, weight, targetSize, names1 = None, names2 = None):
    """
     adjusts the contents of the composite model so as to maximize
        the weighted classification accuracty across the two data sets.
    
        The resulting composite model, with _targetSize_ models, is returned.
    
        **Notes**:
    
          - if _names1_ and _names2_ are not provided, _set1_ and _set2_ should
            have the same ordering of columns and _model_ should have already
            have had _SetInputOrder()_ called.
    
      
    """
