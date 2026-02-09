"""
 command line utility for building composite models

#DOC

**Usage**

  BuildComposite [optional args] filename

Unless indicated otherwise (via command line arguments), _filename_ is
a QDAT file.

**Command Line Arguments**

  - -o *filename*: name of the output file for the pickled composite

  - -n *num*: number of separate models to add to the composite

  - -p *tablename*: store persistence data in the database
     in table *tablename*

  - -N *note*: attach some arbitrary text to the persistence data

  - -b *filename*: name of the text file to hold examples from the
     holdout set which are misclassified

  - -s: split the data into training and hold-out sets before building
     the composite

  - -f *frac*: the fraction of data to use in the training set when the
     data is split

  - -r: randomize the activities (for testing purposes).  This ignores
     the initial distribution of activity values and produces each
     possible activity value with equal likliehood.

  - -S: shuffle the activities (for testing purposes) This produces
     a permutation of the input activity values.

  - -l: locks the random number generator to give consistent sets
     of training and hold-out data.  This is primarily intended
     for testing purposes.

  - -B: use a so-called Bayesian composite model.

  - -d *database name*: instead of reading the data from a QDAT file,
     pull it from a database.  In this case, the _filename_ argument
     provides the name of the database table containing the data set.

  - -D: show a detailed breakdown of the composite model performance
     across the training and, when appropriate, hold-out sets.

  - -P *pickle file name*: write out the pickled data set to the file

  - -F *filter frac*: filters the data before training to change the
     distribution of activity values in the training set.  *filter
     frac* is the fraction of the training set that should have the
     target value.  **See note below on data filtering.**

  - -v *filter value*: filters the data before training to change the
     distribution of activity values in the training set. *filter
     value* is the target value to use in filtering.  **See note below
     on data filtering.**

  - --modelFiltFrac *model filter frac*: Similar to filter frac above,
     in this case the data is filtered for each model in the composite
     rather than a single overall filter for a composite. *model
     filter frac* is the fraction of the training set for each model
     that should have the target value (*model filter value*).

  - --modelFiltVal *model filter value*: target value to use for
     filtering data before training each model in the composite.

  - -t *threshold value*: use high-confidence predictions for the
     final analysis of the hold-out data.

  - -Q *list string*: the values of quantization bounds for the
     activity value.  See the _-q_ argument for the format of *list
     string*.

  - --nRuns *count*: build *count* composite models

  - --prune: prune any models built

  - -h: print a usage message and exit.

  - -V: print the version number and exit

  *-*-*-*-*-*-*-*- Tree-Related Options -*-*-*-*-*-*-*-*

  - -g: be less greedy when training the models.

  - -G *number*: force trees to be rooted at descriptor *number*.

  - -L *limit*: provide an (integer) limit on individual model
     complexity

  - -q *list string*: Add QuantTrees to the composite and use the list
     specified in *list string* as the number of target quantization
     bounds for each descriptor.  Don't forget to include 0's at the
     beginning and end of *list string* for the name and value fields.
     For example, if there are 4 descriptors and you want 2 quant
     bounds apiece, you would use _-q "[0,2,2,2,2,0]"_.
     Two special cases:
       1) If you would like to ignore a descriptor in the model
          building, use '-1' for its number of quant bounds.
       2) If you have integer valued data that should not be quantized
          further, enter 0 for that descriptor.

  - --recycle: allow descriptors to be used more than once in a tree

  - --randomDescriptors=val: toggles growing random forests with val
      randomly-selected descriptors available at each node.


  *-*-*-*-*-*-*-*- KNN-Related Options -*-*-*-*-*-*-*-*

  - --doKnn: use K-Nearest Neighbors models

  - --knnK=*value*: the value of K to use in the KNN models

  - --knnTanimoto: use the Tanimoto metric in KNN models

  - --knnEuclid: use a Euclidean metric in KNN models

  *-*-*-*-*-*-*- Naive Bayes Classifier Options -*-*-*-*-*-*-*-*
  - --doNaiveBayes : use Naive Bayes classifiers

  - --mEstimateVal : the value to be used in the m-estimate formula
      If this is greater than 0.0, we use it to compute the conditional
      probabilities by the m-estimate

  *-*-*-*-*-*-*-*- SVM-Related Options -*-*-*-*-*-*-*-*

  **** NOTE: THESE ARE DISABLED ****

# #   - --doSVM: use Support-vector machines

# #   - --svmKernel=*kernel*: choose the type of kernel to be used for
# #     the SVMs.  Options are:
# #     The default is:

# #   - --svmType=*type*: choose the type of support-vector machine
# #     to be used.  Options are:
# #     The default is:

# #   - --svmGamma=*gamma*: provide the gamma value for the SVMs.  If this
# #     is not provided, a grid search will be carried out to determine an
# #     optimal *gamma* value for each SVM.

# #   - --svmCost=*cost*: provide the cost value for the SVMs.  If this is
# #     not provided, a grid search will be carried out to determine an
# #     optimal *cost* value for each SVM.

# #   - --svmWeights=*weights*: provide the weight values for the
# #     activities.  If provided this should be a sequence of (label,
# #     weight) 2-tuples *nActs* long.  If not provided, a weight of 1
# #     will be used for each activity.

# #   - --svmEps=*epsilon*: provide the epsilon value used to determine
# #     when the SVM has converged.  Defaults to 0.001

# #   - --svmDegree=*degree*: provide the degree of the kernel (when
# #     sensible) Defaults to 3

# #   - --svmCoeff=*coeff*: provide the coefficient for the kernel (when
# #     sensible) Defaults to 0

# #   - --svmNu=*nu*: provide the nu value for the kernel (when sensible)
# #     Defaults to 0.5

# #   - --svmDataType=*float*: if the data is contains only 1 and 0 s, specify by
# #     using binary. Defaults to float

# #   - --svmCache=*cache*: provide the size of the memory cache (in MB)
# #     to be used while building the SVM.  Defaults to 40

**Notes**

  - *Data filtering*: When there is a large disparity between the
    numbers of points with various activity levels present in the
    training set it is sometimes desirable to train on a more
    homogeneous data set.  This can be accomplished using filtering.
    The filtering process works by selecting a particular target
    fraction and target value.  For example, in a case where 95% of
    the original training set has activity 0 and ony 5% activity 1, we
    could filter (by randomly removing points with activity 0) so that
    30% of the data set used to build the composite has activity 1.


"""
from __future__ import annotations
from _warnings import warn
import numpy as numpy
import pickle as pickle
from rdkit import DataStructs
from rdkit.Dbase import DbModule
from rdkit.ML.Composite import BayesComposite
from rdkit.ML.Composite import Composite
from rdkit.ML import CompositeRun
import rdkit.ML.CompositeRun
from rdkit.ML.Data import DataUtils
from rdkit.ML.Data import SplitData
from rdkit.ML import ScreenComposite
from rdkit.utils import listutils
import sys as sys
import time as time
__all__ = ['BayesComposite', 'Composite', 'CompositeRun', 'DataStructs', 'DataUtils', 'DbModule', 'GetCommandLine', 'ParseArgs', 'RunIt', 'RunOnData', 'ScreenComposite', 'SetDefaults', 'ShowVersion', 'SplitData', 'Usage', 'listutils', 'message', 'numpy', 'pickle', 'sys', 'testall', 'time', 'warn']
def GetCommandLine(details):
    """
     #DOC
    
      
    """
def ParseArgs(runDetails):
    """
     parses command line arguments and updates _runDetails_
    
          **Arguments**
    
            - runDetails:  a _CompositeRun.CompositeRun_ object.
    
      
    """
def RunIt(details, progressCallback = None, saveIt = 1, setDescNames = 0):
    """
     does the actual work of building a composite model
    
        **Arguments**
    
          - details:  a _CompositeRun.CompositeRun_ object containing details
            (options, parameters, etc.) about the run
    
          - progressCallback: (optional) a function which is called with a single
            argument (the number of models built so far) after each model is built.
    
          - saveIt: (optional) if this is nonzero, the resulting model will be pickled
            and dumped to the filename specified in _details.outName_
    
          - setDescNames: (optional) if nonzero, the composite's _SetInputOrder()_ method
            will be called using the results of the data set's _GetVarNames()_ method;
            it is assumed that the details object has a _descNames attribute which
            is passed to the composites _SetDescriptorNames()_ method.  Otherwise
            (the default), _SetDescriptorNames()_ gets the results of _GetVarNames()_.
    
        **Returns**
    
          the composite model constructed
    
    
      
    """
def RunOnData(details, data, progressCallback = None, saveIt = 1, setDescNames = 0):
    ...
def SetDefaults(runDetails = None):
    """
      initializes a details object with default values
    
          **Arguments**
    
            - details:  (optional) a _CompositeRun.CompositeRun_ object.
              If this is not provided, the global _runDetails will be used.
    
          **Returns**
    
            the initialized _CompositeRun_ object.
    
    
      
    """
def ShowVersion(includeArgs = 0):
    """
     prints the version number
    
      
    """
def Usage():
    """
     provides a list of arguments for when this is used from the command line
    
      
    """
def message(msg):
    """
     emits messages to _sys.stdout_
        override this in modules which import this one to redirect output
    
        **Arguments**
    
          - msg: the string to be displayed
    
      
    """
def testall(composite, examples, badExamples = list()):
    """
     screens a number of examples past a composite
    
        **Arguments**
    
          - composite: a composite model
    
          - examples: a list of examples (with results) to be screened
    
          - badExamples: a list to which misclassified examples are appended
    
        **Returns**
    
          a list of 2-tuples containing:
    
            1) a vote
    
            2) a confidence
    
          these are the votes and confidence levels for **misclassified** examples
    
      
    """
__VERSION_STRING: str = '3.2.3'
__warningregistry__: dict = {'version': 4}
_runDetails: rdkit.ML.CompositeRun.CompositeRun  # value = <rdkit.ML.CompositeRun.CompositeRun object>
_verbose: int = 1
