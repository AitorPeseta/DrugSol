"""

MolVS - Molecule Validation and Standardization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MolVS is a python tool built on top of RDKit that performs validation and standardization of chemical structures.

Note that the C++ reimplementation of this is available in the module rdkit.Chem.MolStandardize.rdMolStandardize

:copyright: (c) 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
from __future__ import annotations
import logging as logging
from rdkit import Chem
from rdkit.Chem.MolStandardize.errors import MolVSError
from rdkit.Chem.MolStandardize.errors import StandardizeError
from rdkit.Chem.MolStandardize.errors import ValidateError
from rdkit.Chem.MolStandardize.standardize import Standardizer
from rdkit.Chem.MolStandardize.standardize import canonicalize_tautomer_smiles
from rdkit.Chem.MolStandardize.standardize import enumerate_tautomers_smiles
from rdkit.Chem.MolStandardize.standardize import standardize_smiles
from rdkit.Chem.MolStandardize.validate import Validator
from rdkit.Chem.MolStandardize.validate import validate_smiles
from .charge import *
from .errors import *
from .fragment import *
from .metal import *
from .normalize import *
from .rdMolStandardize import *
from .standardize import *
from .tautomer import *
from .utils import *
from .validate import *
from .validations import *
__all__ = ['Chem', 'MolVSError', 'ReorderTautomers', 'StandardizeError', 'Standardizer', 'ValidateError', 'Validator', 'canonicalize_tautomer_smiles', 'charge', 'enumerate_tautomers_smiles', 'errors', 'fragment', 'log', 'logging', 'metal', 'normalize', 'rdMolStandardize', 'standardize', 'standardize_smiles', 'tautomer', 'utils', 'validate', 'validate_smiles', 'validations']
def ReorderTautomers(molecule):
    """
    Returns the list of the molecule's tautomers
        so that the canonical one as determined by the canonical
        scoring system in TautomerCanonicalizer appears first.
    
        :param molecule: An RDKit Molecule object.
        :return: A list of Molecule objects.
        
    """
__author__: str = 'Matt Swain'
__copyright__: str = 'Copyright 2016 Matt Swain'
__email__: str = 'm.swain@me.com'
__license__: str = 'MIT'
__title__: str = 'MolVS'
__version__: str = '0.1.1'
__warningregistry__: dict = {'version': 4}
log: logging.Logger  # value = <Logger rdkit.Chem.MolStandardize (WARNING)>
