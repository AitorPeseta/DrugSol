"""
 command line utility to report on the contributions of descriptors to
tree-based composite models

Usage:  AnalyzeComposite [optional args] <models>

      <models>: file name(s) of pickled composite model(s)
        (this is the name of the db table if using a database)

    Optional Arguments:

      -n number: the number of levels of each model to consider

      -d dbname: the database from which to read the models

      -N Note: the note string to search for to pull models from the database

      -v: be verbose whilst screening
"""
from __future__ import annotations
from _warnings import warn
import numpy as numpy
import pickle as pickle
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.ML.Data import Stats
from rdkit.ML.DecTree import Tree
from rdkit.ML.DecTree import TreeUtils
from rdkit.ML import ScreenComposite
import sys as sys
__all__ = ['DbConnect', 'ErrorStats', 'ProcessIt', 'ScreenComposite', 'ShowStats', 'Stats', 'Tree', 'TreeUtils', 'Usage', 'numpy', 'pickle', 'sys', 'warn']
def ErrorStats(conn, where, enrich = 1):
    ...
def ProcessIt(composites, nToConsider = 3, verbose = 0):
    ...
def ShowStats(statD, enrich = 1):
    ...
def Usage():
    ...
__VERSION_STRING: str = '2.2.0'
__warningregistry__: dict = {'version': 4}
