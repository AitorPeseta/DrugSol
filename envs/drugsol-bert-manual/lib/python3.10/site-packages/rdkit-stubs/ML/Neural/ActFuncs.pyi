"""
 Activation functions for neural network nodes

Activation functions should implement the following API:

 - _Eval(x)_: returns the value of the function at a given point

 - _Deriv(x)_: returns the derivative of the function at a given point

The current Backprop implementation also requires:

 - _DerivFromVal(val)_: returns the derivative of the function when its
                        value is val

In all cases _x_ is a float as is the value returned.

"""
from __future__ import annotations
import math as math
__all__ = ['ActFunc', 'Sigmoid', 'TanH', 'math']
class ActFunc:
    """
     "virtual base class" for activation functions
    
      
    """
    def __call__(self, x):
        ...
class Sigmoid(ActFunc):
    """
     the standard sigmoidal function 
    """
    def Deriv(self, x):
        ...
    def DerivFromVal(self, val):
        ...
    def Eval(self, x):
        ...
    def __init__(self, beta = 1.0):
        ...
class TanH(ActFunc):
    """
     the standard hyperbolic tangent function 
    """
    def Deriv(self, x):
        ...
    def DerivFromVal(self, val):
        ...
    def Eval(self, x):
        ...
    def __init__(self, beta = 1.0):
        ...
