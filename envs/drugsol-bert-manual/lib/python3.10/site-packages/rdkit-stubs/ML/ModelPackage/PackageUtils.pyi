from __future__ import annotations
from _elementtree import SubElement
import time as time
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import ElementTree
__all__ = ['Element', 'ElementTree', 'PackageToXml', 'SubElement', 'time']
def PackageToXml(pkg, summary = 'N/A', trainingDataId = 'N/A', dataPerformance = list(), recommendedThreshold = None, classDescriptions = None, modelType = None, modelOrganism = None):
    """
     generates XML for a package that follows the RD_Model.dtd
    
        If provided, dataPerformance should be a sequence of 2-tuples:
          ( note, performance )
        where performance is of the form:
          ( accuracy, avgCorrectConf, avgIncorrectConf, confusionMatrix, thresh, avgSkipConf, nSkipped )
          the last four elements are optional
    
        
    """
def _ConvertModelPerformance(perf, modelPerf):
    ...
