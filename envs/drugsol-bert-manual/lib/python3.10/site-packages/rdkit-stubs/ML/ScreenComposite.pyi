"""
 command line utility for screening composite models

**Usage**

  _ScreenComposite [optional args] modelfile(s) datafile_

Unless indicated otherwise (via command line arguments), _modelfile_ is
a file containing a pickled composite model and _filename_ is a QDAT file.

**Command Line Arguments**

  - -t *threshold value(s)*: use high-confidence predictions for the final
     analysis of the hold-out data.  The threshold value can be either a single
     float or a list/tuple of floats.  All thresholds should be between
     0.0 and 1.0

  - -D: do a detailed screen.

  - -d *database name*: instead of reading the data from a QDAT file,
     pull it from a database.  In this case, the _datafile_ argument
     provides the name of the database table containing the data set.

  - -N *note*: use all models from the database which have this note.
               The modelfile argument should contain the name of the table
               with the models.

  - -H: screen only the hold out set (works only if a version of
        BuildComposite more recent than 1.2.2 was used).

  - -T: screen only the training set (works only if a version of
        BuildComposite more recent than 1.2.2 was used).

  - -E: do a detailed Error analysis.  This shows each misclassified
     point and the number of times it was missed across all screened
     composites.  If the --enrich argument is also provided, only compounds
     that have true activity value equal to the enrichment value will be
     used.

  - --enrich *enrichVal*: target "active" value to be used in calculating
     enrichments.

  - -A: show All predictions.

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

  - -V: be verbose when screening multiple models

  - -h: show this message and exit

  - --OOB: Do out an "out-of-bag" generalization error estimate.  This only
      makes sense when applied to the original data set.

  - --pickleCol *colId*: index of the column containing a pickled value
      (used primarily for cases where fingerprints are used as descriptors)

  *** Options for making Prediction (Hanneke) Plots ***

  - --predPlot=<fileName>: triggers the generation of a Hanneke plot and
      sets the name of the .txt file which will hold the output data.
      A Gnuplot control file, <fileName>.gnu, will also be generated.

  - --predActTable=<name> (optional):  name of the database table
      containing activity values.  If this is not provided, activities
      will be read from the same table containing the screening data

  - --predActCol=<name> (optional):  name of the activity column. If not
      provided, the name of the last column in the activity table will
      be used.

  - --predLogScale (optional):  If provided, the x axis of the
      prediction plot (the activity axis) will be plotted using a log
      scale

  - --predShow: launch a gnuplot instance and display the prediction
      plot (the plot will still be written to disk).

  *** The following options are likely obsolete ***

  - -P: read pickled data.  The datafile argument should contain
     a pickled data set. *relevant only to qdat files*

  - -q: data are not quantized (the composite should take care of
     quantization itself if it requires quantized data). *relevant only to
     qdat files*



"""
from PIL import Image
from PIL import ImageDraw
from __future__ import annotations
from _warnings import warn
import numpy as numpy
import os as os
import pickle as pickle
from rdkit import DataStructs
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.Dbase import DbModule
from rdkit.ML import CompositeRun
import rdkit.ML.CompositeRun
from rdkit.ML.Data import DataUtils
from rdkit.ML.Data import SplitData
import sys as sys
__all__ = ['CalcEnrichment', 'CollectResults', 'CompositeRun', 'DataStructs', 'DataUtils', 'DbConnect', 'DbModule', 'DetailedScreen', 'GetScreenImage', 'Go', 'Image', 'ImageDraw', 'MakePredPlot', 'ParseArgs', 'PrepareDataFromDetails', 'ScreenFromDetails', 'ScreenIt', 'ScreenToHtml', 'SetDefaults', 'ShowVersion', 'ShowVoteResults', 'SplitData', 'Usage', 'error', 'hasPil', 'message', 'numpy', 'os', 'pickle', 'sys', 'warn']
def CalcEnrichment(mat, tgt = 1):
    ...
def CollectResults(indices, dataSet, composite, callback = None, appendExamples = 0, errorEstimate = 0):
    """
     screens a set of examples through a composite and returns the
      results
    #DOC
    
      **Arguments**
    
        - examples: the examples to be screened (a sequence of sequences)
           it's assumed that the last element in each example is it's "value"
    
        - composite:  the composite model to be used
    
        - callback: (optional)  if provided, this should be a function
          taking a single argument that is called after each example is
          screened with the number of examples screened so far as the
          argument.
    
        - appendExamples: (optional)  this value is passed on to the
          composite's _ClassifyExample()_ method.
    
        - errorEstimate: (optional) calculate the "out of bag" error
          estimate for the composite using Breiman's definition.  This
          only makes sense when screening the original data set!
          [L. Breiman "Out-of-bag Estimation", UC Berkeley Dept of
          Statistics Technical Report (1996)]
    
      **Returns**
    
        a list of 3-tuples _nExamples_ long:
    
          1)  answer: the value from the example
    
          2)  pred: the composite model's prediction
    
          3)  conf: the confidence of the composite
    
      
    """
def DetailedScreen(indices, data, composite, threshold = 0, screenResults = None, goodVotes = None, badVotes = None, noVotes = None, callback = None, appendExamples = 0, errorEstimate = 0):
    """
     screens a set of examples cross a composite and breaks the
          predictions into *correct*,*incorrect* and *unclassified* sets.
    #DOC
      **Arguments**
    
        - examples: the examples to be screened (a sequence of sequences)
           it's assumed that the last element in each example is its "value"
    
        - composite:  the composite model to be used
    
        - threshold: (optional) the threshold to be used to decide whether
          or not a given prediction should be kept
    
        - screenResults: (optional) the results of screening the results
          (a sequence of 3-tuples in the format returned by
          _CollectResults()_).  If this is provided, the examples will not
          be screened again.
    
        - goodVotes,badVotes,noVotes: (optional)  if provided these should
          be lists (or anything supporting an _append()_ method) which
          will be used to pass the screening results back.
    
        - callback: (optional)  if provided, this should be a function
          taking a single argument that is called after each example is
          screened with the number of examples screened so far as the
          argument.
    
        - appendExamples: (optional)  this value is passed on to the
          composite's _ClassifyExample()_ method.
    
        - errorEstimate: (optional) calculate the "out of bag" error
          estimate for the composite using Breiman's definition.  This
          only makes sense when screening the original data set!
          [L. Breiman "Out-of-bag Estimation", UC Berkeley Dept of
          Statistics Technical Report (1996)]
    
      **Notes**
    
        - since this function doesn't return anything, if one or more of
          the arguments _goodVotes_, _badVotes_, and _noVotes_ is not
          provided, there's not much reason to call it
    
      
    """
def GetScreenImage(nGood, nBad, nRej, size = None):
    ...
def Go(details):
    ...
def MakePredPlot(details, indices, data, goodVotes, badVotes, nRes, idCol = 0, verbose = 0):
    """
    
    
      **Arguments**
    
        - details:  a CompositeRun.RunDetails object
    
        - indices: a sequence of integer indices into _data_
    
        - data: the data set in question.  We assume that the ids for
          the data points are in the _idCol_ column
    
        - goodVotes/badVotes: predictions where the model was correct/incorrect.
          These are sequences of 4-tuples:
            (answer,prediction,confidence,index into _indices_)
    
      
    """
def ParseArgs(details):
    ...
def PrepareDataFromDetails(model, details, data, verbose = 0):
    ...
def ScreenFromDetails(models, details, callback = None, setup = None, appendExamples = 0, goodVotes = None, badVotes = None, noVotes = None, data = None, enrichments = None):
    """
      Screens a set of data using a a _CompositeRun.CompositeRun_
           instance to provide parameters
    
    # DOC
    
        The actual data to be used are extracted from the database and
        table specified in _details_
    
        Aside from dataset construction,  _ShowVoteResults()_ does most of
        the heavy lifting here.
    
      **Arguments**
    
          - model: a composite model
    
          - details:  a _CompositeRun.CompositeRun_ object containing details
            (options, parameters, etc.) about the run
    
          - callback: (optional)  if provided, this should be a function
            taking a single argument that is called after each example is
            screened with the number of examples screened so far as the
            argument.
    
          - setup: (optional) a function taking a single argument which is
            called at the start of screening with the number of points to
            be screened as the argument.
    
          - appendExamples: (optional)  this value is passed on to the
            composite's _ClassifyExample()_ method.
    
          - goodVotes,badVotes,noVotes: (optional)  if provided these should
            be lists (or anything supporting an _append()_ method) which
            will be used to pass the screening results back.
    
    
      **Returns**
    
        a 7-tuple:
    
          1) the number of good (correct) predictions
    
          2) the number of bad (incorrect) predictions
    
          3) the number of predictions skipped due to the _threshold_
    
          4) the average confidence in the good predictions
    
          5) the average confidence in the bad predictions
    
          6) the average confidence in the skipped predictions
    
          7) the results table
    
      
    """
def ScreenIt(composite, indices, data, partialVote = 0, voteTol = 0.0, verbose = 1, screenResults = None, goodVotes = None, badVotes = None, noVotes = None):
    """
     screens a set of data using a composite model and prints out
                 statistics about the screen.
    #DOC
        The work of doing the screening and processing the results is
        handled by _DetailedScreen()_
    
      **Arguments**
    
        - composite:  the composite model to be used
    
        - data: the examples to be screened (a sequence of sequences)
           it's assumed that the last element in each example is its "value"
    
        - partialVote: (optional) toggles use of the threshold value in
          the screnning.
    
        - voteTol: (optional) the threshold to be used to decide whether or not a
          given prediction should be kept
    
        - verbose: (optional) sets degree of verbosity of the screening
    
        - screenResults: (optional) the results of screening the results
          (a sequence of 3-tuples in the format returned by
          _CollectResults()_).  If this is provided, the examples will not
          be screened again.
    
        - goodVotes,badVotes,noVotes: (optional)  if provided these should
          be lists (or anything supporting an _append()_ method) which
          will be used to pass the screening results back.
    
    
      **Returns**
    
        a 7-tuple:
    
          1) the number of good (correct) predictions
    
          2) the number of bad (incorrect) predictions
    
          3) the number of predictions skipped due to the _threshold_
    
          4) the average confidence in the good predictions
    
          5) the average confidence in the bad predictions
    
          6) the average confidence in the skipped predictions
    
          7) None
    
      
    """
def ScreenToHtml(nGood, nBad, nRej, avgGood, avgBad, avgSkip, voteTable, imgDir = '.', fullPage = 1, skipImg = 0, includeDefs = 1):
    """
     returns the text of a web page showing the screening details
    #DOC
        **Arguments**
    
         - nGood: number of correct predictions
    
         - nBad:  number of incorrect predictions
    
         - nRej:  number of rejected predictions
    
         - avgGood: average correct confidence
    
         - avgBad:  average incorrect confidence
    
         - avgSkip: average rejected confidence
    
         - voteTable: vote table
    
         - imgDir: (optional) the directory to be used to hold the vote
           image (if constructed)
    
       **Returns**
    
         a string containing HTML
    
      
    """
def SetDefaults(details = None):
    ...
def ShowVersion(includeArgs = 0):
    """
     prints the version number of the program
    
      
    """
def ShowVoteResults(indices, data, composite, nResultCodes, threshold, verbose = 1, screenResults = None, callback = None, appendExamples = 0, goodVotes = None, badVotes = None, noVotes = None, errorEstimate = 0):
    """
     screens the results and shows a detailed workup
    
      The work of doing the screening and processing the results is
      handled by _DetailedScreen()_
    #DOC
    
      **Arguments**
    
        - examples: the examples to be screened (a sequence of sequences)
           it's assumed that the last element in each example is its "value"
    
        - composite:  the composite model to be used
    
        - nResultCodes: the number of possible results the composite can
          return
    
        - threshold: the threshold to be used to decide whether or not a
          given prediction should be kept
    
        - screenResults: (optional) the results of screening the results
          (a sequence of 3-tuples in the format returned by
          _CollectResults()_).  If this is provided, the examples will not
          be screened again.
    
        - callback: (optional)  if provided, this should be a function
          taking a single argument that is called after each example is
          screened with the number of examples screened so far as the
          argument.
    
        - appendExamples: (optional)  this value is passed on to the
          composite's _ClassifyExample()_ method.
    
        - goodVotes,badVotes,noVotes: (optional)  if provided these should
          be lists (or anything supporting an _append()_ method) which
          will be used to pass the screening results back.
    
        - errorEstimate: (optional) calculate the "out of bag" error
          estimate for the composite using Breiman's definition.  This
          only makes sense when screening the original data set!
          [L. Breiman "Out-of-bag Estimation", UC Berkeley Dept of
          Statistics Technical Report (1996)]
    
      **Returns**
    
        a 7-tuple:
    
          1) the number of good (correct) predictions
    
          2) the number of bad (incorrect) predictions
    
          3) the number of predictions skipped due to the _threshold_
    
          4) the average confidence in the good predictions
    
          5) the average confidence in the bad predictions
    
          6) the average confidence in the skipped predictions
    
          7) the results table
    
      
    """
def Usage():
    """
     prints a list of arguments for when this is used from the
      command line and then exits
    
      
    """
def _processVoteList(votes, data):
    """
     *Internal Use Only*
    
      converts a list of 4 tuples: (answer,prediction,confidence,idx) into
      an alternate list: (answer,prediction,confidence,data point)
    
       **Arguments**
    
         - votes: a list of 4 tuples: (answer, prediction, confidence,
           index)
    
         - data: a _DataUtils.MLData.MLDataSet_
    
    
       **Note**: alterations are done in place in the _votes_ list
    
      
    """
def error(msg):
    """
     emits messages to _sys.stderr_
        override this in modules which import this one to redirect output
    
        **Arguments**
    
          - msg: the string to be displayed
    
      
    """
def message(msg, noRet = 0):
    """
     emits messages to _sys.stdout_
        override this in modules which import this one to redirect output
    
        **Arguments**
    
          - msg: the string to be displayed
    
      
    """
__VERSION_STRING: str = '3.3.0'
__warningregistry__: dict = {'version': 4}
_details: rdkit.ML.CompositeRun.CompositeRun  # value = <rdkit.ML.CompositeRun.CompositeRun object>
hasPil: int = 1
