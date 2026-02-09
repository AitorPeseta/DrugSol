"""

molvs.validate
~~~~~~~~~~~~~~

This module contains the main :class:`~molvs.validate.Validator` class that can be used to perform all
:class:`Validations <molvs.validations.Validation>`, as well as the :func:`~molvs.validate.validate_smiles()`
convenience function.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
from __future__ import annotations
from _warnings import warn
import logging as logging
from rdkit import Chem
from rdkit.Chem.MolStandardize.errors import StopValidateError
import rdkit.Chem.MolStandardize.validations
import sys as sys
__all__ = ['Chem', 'LONG_FORMAT', 'LogHandler', 'SIMPLE_FORMAT', 'StopValidateError', 'VALIDATIONS', 'Validator', 'logging', 'sys', 'validate_smiles', 'warn']
class LogHandler(logging.Handler):
    """
    A simple logging Handler that just stores logs in an array until flushed.
    """
    def __init__(self):
        ...
    def close(self):
        """
        Close the handler.
        """
    def emit(self, record):
        """
        Append the record.
        """
    def flush(self):
        """
        Clear the log records.
        """
    @property
    def logmessages(self):
        ...
class Validator:
    """
    The main class for running :class:`Validations <molvs.validations.Validation>` on molecules.
    """
    def __call__(self, mol):
        """
        Calling a Validator instance like a function is the same as calling its
                :meth:`~molvs.validate.Validator.validate` method.
        """
    def __init__(self, validations = (rdkit.Chem.MolStandardize.validations.IsNoneValidation, rdkit.Chem.MolStandardize.validations.NoAtomValidation, rdkit.Chem.MolStandardize.validations.FragmentValidation, rdkit.Chem.MolStandardize.validations.NeutralValidation, rdkit.Chem.MolStandardize.validations.IsotopeValidation), log_format = '%(levelname)s: [%(validation)s] %(message)s', level = 20, stdout = False, raw = False):
        """
        Initialize a Validator with the following parameters:
        
                :param validations: A list of Validations to apply (default: :data:`~molvs.validations.VALIDATIONS`).
                :param string log_format: A string format (default: :data:`~molvs.validate.SIMPLE_FORMAT`).
                :param level: The minimum logging level to output.
                :param bool stdout: Whether to send log messages to standard output.
                :param bool raw: Whether to return raw :class:`~logging.LogRecord` objects instead of formatted log strings.
                
        """
    def validate(self, mol):
        """
        """
def validate_smiles(smiles):
    """
    Return log messages for a given SMILES string using the default validations.
    
        Note: This is a convenience function for quickly validating a single SMILES string. It is more efficient to use
        the :class:`~molvs.validate.Validator` class directly when working with many molecules or when custom options
        are needed.
    
        :param string smiles: The SMILES for the molecule.
        :returns: A list of log messages.
        :rtype: list of strings.
        
    """
LONG_FORMAT: str = '%(asctime)s - %(levelname)s - %(validation)s - %(message)s'
SIMPLE_FORMAT: str = '%(levelname)s: [%(validation)s] %(message)s'
VALIDATIONS: tuple = (rdkit.Chem.MolStandardize.validations.IsNoneValidation, rdkit.Chem.MolStandardize.validations.NoAtomValidation, rdkit.Chem.MolStandardize.validations.FragmentValidation, rdkit.Chem.MolStandardize.validations.NeutralValidation, rdkit.Chem.MolStandardize.validations.IsotopeValidation)
__warningregistry__: dict = {'version': 4}
