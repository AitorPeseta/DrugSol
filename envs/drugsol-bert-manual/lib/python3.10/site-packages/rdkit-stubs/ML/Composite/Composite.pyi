"""
 code for dealing with composite models

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
import pickle as pickle
from rdkit.ML.Data import DataUtils
__all__ = ['Composite', 'DataUtils', 'numpy', 'pickle']
class Composite:
    """
    a composite model
    
    
        **Notes**
    
        - adding a model which is already present just results in its count
           field being incremented and the errors being averaged.
    
        - typical usage:
    
           1) grow the composite with AddModel until happy with it
    
           2) call AverageErrors to calculate the average error values
    
           3) call SortModels to put things in order by either error or count
    
        - Composites can support individual models requiring either quantized or
           nonquantized data.  This is done by keeping a set of quantization bounds
           (_QuantBounds_) in the composite and quantizing data passed in when required.
           Quantization bounds can be set and interrogated using the
           _Get/SetQuantBounds()_ methods.  When models are added to the composite,
           it can be indicated whether or not they require quantization.
    
        - Composites are also capable of extracting relevant variables from longer lists.
          This is accessible using _SetDescriptorNames()_ to register the descriptors about
          which the composite cares and _SetInputOrder()_ to tell the composite what the
          ordering of input vectors will be.  **Note** there is a limitation on this: each
          model needs to take the same set of descriptors as inputs.  This could be changed.
    
      
    """
    def AddModel(self, model, error, needsQuantization = 1):
        """
         Adds a model to the composite
        
             **Arguments**
        
               - model: the model to be added
        
               - error: the model's error
        
               - needsQuantization: a toggle to indicate whether or not this model
                  requires quantized inputs
        
             **NOTE**
        
                - this can be used as an alternative to _Grow()_ if you already have
                  some models constructed
        
                - the errList is run as an accumulator,
                  you probably want to call _AverageErrors_ after finishing the forest
        
            
        """
    def AverageErrors(self):
        """
         convert local summed error to average error
        
            
        """
    def ClassifyExample(self, example, threshold = 0, appendExample = 0, onlyModels = None):
        """
         classifies the given example using the entire composite
        
              **Arguments**
        
               - example: the data to be classified
        
               - threshold:  if this is a number greater than zero, then a
                  classification will only be returned if the confidence is
                  above _threshold_.  Anything lower is returned as -1.
        
               - appendExample: toggles saving the example on the models
        
               - onlyModels: if provided, this should be a sequence of model
                 indices. Only the specified models will be used in the
                 prediction.
        
              **Returns**
        
                a (result,confidence) tuple
        
        
              **FIX:**
                statistics sucks... I'm not seeing an obvious way to get
                   the confidence intervals.  For that matter, I'm not seeing
                   an unobvious way.
        
                For now, this is just treated as a voting problem with the confidence
                measure being the percent of models which voted for the winning result.
        
            
        """
    def ClearModelExamples(self):
        ...
    def CollectVotes(self, example, quantExample, appendExample = 0, onlyModels = None):
        """
         collects votes across every member of the composite for the given example
        
             **Arguments**
        
               - example: the example to be voted upon
        
               - quantExample: the quantized form of the example
        
               - appendExample: toggles saving the example on the models
        
               - onlyModels: if provided, this should be a sequence of model
                 indices. Only the specified models will be used in the
                 prediction.
        
             **Returns**
        
               a list with a vote from each member
        
            
        """
    def GetActivityQuantBounds(self):
        ...
    def GetAllData(self):
        """
         Returns everything we know
        
              **Returns**
        
                a 3-tuple consisting of:
        
                  1) our list of models
        
                  2) our list of model counts
        
                  3) our list of model errors
        
            
        """
    def GetCount(self, i):
        """
         returns the count of the _i_th model
        
            
        """
    def GetDataTuple(self, i):
        """
         returns all relevant data about a particular model
        
              **Arguments**
        
                i: an integer indicating which model should be returned
        
              **Returns**
        
                a 3-tuple consisting of:
        
                  1) the model
        
                  2) its count
        
                  3) its error
            
        """
    def GetDescriptorNames(self):
        """
         returns the names of the descriptors this composite uses
        
            
        """
    def GetError(self, i):
        """
         returns the error of the _i_th model
        
            
        """
    def GetInputOrder(self):
        """
         returns the input order (used in remapping inputs)
        
            
        """
    def GetModel(self, i):
        """
         returns a particular model
        
            
        """
    def GetQuantBounds(self):
        """
         returns the quantization bounds
        
             **Returns**
        
               a 2-tuple consisting of:
        
                 1) the list of quantization bounds
        
                 2) the nPossibleVals list
        
            
        """
    def GetVoteDetails(self):
        """
         returns the votes from the last classification
        
              This will be _None_ if nothing has yet be classified
            
        """
    def Grow(self, examples, attrs, nPossibleVals, buildDriver, pruner = None, nTries = 10, pruneIt = 0, needsQuantization = 1, progressCallback = None, **buildArgs):
        """
         Grows the composite
        
              **Arguments**
        
               - examples: a list of examples to be used in training
        
               - attrs: a list of the variables to be used in training
        
               - nPossibleVals: this is used to provide a list of the number
                  of possible values for each variable.  It is used if the
                  local quantBounds have not been set (for example for when you
                  are working with data which is already quantized).
        
               - buildDriver: the function to call to build the new models
        
               - pruner: a function used to "prune" (reduce the complexity of)
                  the resulting model.
        
               - nTries: the number of new models to add
        
               - pruneIt: toggles whether or not pruning is done
        
               - needsQuantization: used to indicate whether or not this type of model
                  requires quantized data
        
               - **buildArgs: all other keyword args are passed to _buildDriver_
        
              **Note**
        
                - new models are *added* to the existing ones
        
            
        """
    def MakeHistogram(self):
        """
         creates a histogram of error/count pairs
        
             **Returns**
        
               the histogram as a series of (error, count) 2-tuples
        
            
        """
    def Pickle(self, fileName = 'foo.pkl', saveExamples = 0):
        """
         Writes this composite off to a file so that it can be easily loaded later
        
             **Arguments**
        
               - fileName: the name of the file to be written
        
               - saveExamples: if this is zero, the individual models will have
                 their stored examples cleared.
        
            
        """
    def QuantizeActivity(self, example, activityQuant = None, actCol = -1):
        ...
    def QuantizeExample(self, example, quantBounds = None):
        """
         quantizes an example
        
              **Arguments**
        
               - example: a data point (list, tuple or numpy array)
        
               - quantBounds:  a list of quantization bounds, each quantbound is a
                     list of boundaries.  If this argument is not provided, the composite
                     will use its own quantBounds
        
              **Returns**
        
                the quantized example as a list
        
              **Notes**
        
                - If _example_ is different in length from _quantBounds_, this will
                   assert out.
        
                - This is primarily intended for internal use
        
            
        """
    def SetActivityQuantBounds(self, bounds):
        ...
    def SetCount(self, i, val):
        """
         sets the count of the _i_th model
        
            
        """
    def SetDataTuple(self, i, tup):
        """
         sets all relevant data for a particular tree in the forest
        
              **Arguments**
        
                - i: an integer indicating which model should be returned
        
                - tup: a 3-tuple consisting of:
        
                  1) the model
        
                  2) its count
        
                  3) its error
        
              **Note**
        
                This is included for the sake of completeness, but you need to be
                *very* careful when you use it.
        
            
        """
    def SetDescriptorNames(self, names):
        """
         registers the names of the descriptors this composite uses
        
              **Arguments**
        
               - names: a list of descriptor names (strings).
        
              **NOTE**
        
                 the _names_ list is not
                 copied, so if you modify it later, the composite itself will also be modified.
        
            
        """
    def SetError(self, i, val):
        """
         sets the error of the _i_th model
        
            
        """
    def SetInputOrder(self, colNames):
        """
         sets the input order
        
              **Arguments**
        
                - colNames: a list of the names of the data columns that will be passed in
        
              **Note**
        
                - you must call _SetDescriptorNames()_ first for this to work
        
                - if the local descriptor names do not appear in _colNames_, this will
                  raise an _IndexError_ exception.
            
        """
    def SetModel(self, i, val):
        """
         replaces a particular model
        
              **Note**
        
                This is included for the sake of completeness, but you need to be
                *very* careful when you use it.
        
            
        """
    def SetModelFilterData(self, modelFilterFrac = 0.0, modelFilterVal = 0.0):
        ...
    def SetQuantBounds(self, qBounds, nPossible = None):
        """
         sets the quantization bounds that the composite will use
        
              **Arguments**
        
               - qBounds:  a list of quantization bounds, each quantbound is a
                     list of boundaries
        
               - nPossible:  a list of integers indicating how many possible values
                  each descriptor can take on.
        
              **NOTE**
        
                 - if the two lists are of different lengths, this will assert out
        
                 - neither list is copied, so if you modify it later, the composite
                   itself will also be modified.
        
            
        """
    def SortModels(self, sortOnError = True):
        """
         sorts the list of models
        
              **Arguments**
        
                sortOnError: toggles sorting on the models' errors rather than their counts
        
        
            
        """
    def _RemapInput(self, inputVect):
        """
         remaps the input so that it matches the expected internal ordering
        
              **Arguments**
        
                - inputVect: the input to be reordered
        
              **Returns**
        
                - a list with the reordered (and possible shorter) data
        
              **Note**
        
                - you must call _SetDescriptorNames()_ and _SetInputOrder()_ for this to work
        
                - this is primarily intended for internal use
        
            
        """
    def __getitem__(self, which):
        """
         allows composite[i] to work, returns the data tuple
        
            
        """
    def __init__(self):
        ...
    def __len__(self):
        """
         allows len(composite) to work
        
            
        """
    def __str__(self):
        """
         returns a string representation of the composite
        
            
        """
