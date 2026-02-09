"""
Command line tool to construct an enrichment plot from saved composite models

Usage:  EnrichPlot [optional args] -d dbname -t tablename <models>

Required Arguments:
  -d "dbName": the name of the database for screening

  -t "tablename": provide the name of the table with the data to be screened

  <models>: file name(s) of pickled composite model(s).
     If the -p argument is also provided (see below), this argument is ignored.

Optional Arguments:
  - -a "list": the list of result codes to be considered active.  This will be
        eval'ed, so be sure that it evaluates as a list or sequence of
        integers. For example, -a "[1,2]" will consider activity values 1 and 2
        to be active

  - --enrich "list": identical to the -a argument above.

  - --thresh: sets a threshold for the plot.  If the confidence falls below
          this value, picking will be terminated

  - -H: screen only the hold out set (works only if a version of
        BuildComposite more recent than 1.2.2 was used).

  - -T: screen only the training set (works only if a version of
        BuildComposite more recent than 1.2.2 was used).

  - -S: shuffle activity values before screening

  - -R: randomize activity values before screening

  - -F *filter frac*: filters the data before training to change the
     distribution of activity values in the training set.  *filter frac*
     is the fraction of the training set that should have the target value.
     **See note in BuildComposite help about data filtering**

  - -v *filter value*: filters the data before training to change the
     distribution of activity values in the training set. *filter value*
     is the target value to use in filtering.
     **See note in BuildComposite help about data filtering**

  - -p "tableName": provides the name of a db table containing the
      models to be screened.  If you use this argument, you should also
      use the -N argument (below) to specify a note value.

  - -N "note": provides a note to be used to pull models from a db table.

  - --plotFile "filename": writes the data to an output text file (filename.dat)
    and creates a gnuplot input file (filename.gnu) to plot it

  - --showPlot: causes the gnuplot plot constructed using --plotFile to be
    displayed in gnuplot.

"""
from __future__ import annotations
from _warnings import warn
import numpy as numpy
import pickle as pickle
from rdkit import DataStructs
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML import CompositeRun
from rdkit.ML.Data import DataUtils
from rdkit.ML.Data import SplitData
from rdkit.ML.Data import Stats
from rdkit import RDConfig
import sys as sys
__all__ = ['AccumulateCounts', 'CompositeRun', 'DataStructs', 'DataUtils', 'DbConnect', 'MakePlot', 'RDConfig', 'ScreenModel', 'SplitData', 'Stats', 'Usage', 'cmp', 'error', 'message', 'numpy', 'pickle', 'sys', 'warn']
def AccumulateCounts(predictions, thresh = 0, sortIt = 1):
    """
      Accumulates the data for the enrichment plot for a single model
    
          **Arguments**
    
            - predictions: a list of 3-tuples (as returned by _ScreenModels_)
    
            - thresh: a threshold for the confidence level.  Anything below
              this threshold will not be considered
    
            - sortIt: toggles sorting on confidence levels
    
    
          **Returns**
    
            - a list of 3-tuples:
    
              - the id of the active picked here
    
              - num actives found so far
    
              - number of picks made so far
    
        
    """
def MakePlot(details, final, counts, pickVects, nModels, nTrueActs = -1):
    ...
def ScreenModel(mdl, descs, data, picking = [1], indices = list(), errorEstimate = 0):
    """
     collects the results of screening an individual composite model that match
          a particular value
    
         **Arguments**
    
           - mdl: the composite model
    
           - descs: a list of descriptor names corresponding to the data set
    
           - data: the data set, a list of points to be screened.
    
           - picking: (Optional) a list of values that are to be collected.
             For examples, if you want an enrichment plot for picking the values
             1 and 2, you'd having picking=[1,2].
    
          **Returns**
    
            a list of 4-tuples containing:
    
               - the id of the point
    
               - the true result (from the data set)
    
               - the predicted result
    
               - the confidence value for the prediction
    
        
    """
def Usage():
    """
     displays a usage message and exits 
    """
def cmp(t1, t2):
    ...
def error(msg, dest = ...):
    """
     emits messages to _sys.stderr_
          override this in modules which import this one to redirect output
    
          **Arguments**
    
            - msg: the string to be displayed
    
        
    """
def message(msg, noRet = 0, dest = ...):
    """
     emits messages to _sys.stderr_
          override this in modules which import this one to redirect output
    
          **Arguments**
    
            - msg: the string to be displayed
    
        
    """
__VERSION_STRING: str = '2.4.0'
__warningregistry__: dict = {'version': 4}
