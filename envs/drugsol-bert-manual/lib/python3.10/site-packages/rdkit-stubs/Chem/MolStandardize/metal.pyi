"""

molvs.metal
~~~~~~~~~~~

This module contains tools for disconnecting metal atoms that are defined as covalently bonded to non-metals.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
from __future__ import annotations
from _warnings import warn
import logging as logging
from rdkit import Chem
__all__ = ['Chem', 'MetalDisconnector', 'log', 'logging', 'warn']
class MetalDisconnector:
    """
    Class for breaking covalent bonds between metals and organic atoms under certain conditions.
    """
    def __call__(self, mol):
        """
        Calling a MetalDisconnector instance like a function is the same as calling its disconnect(mol) method.
        """
    def __init__(self):
        ...
    def disconnect(self, mol):
        """
        Break covalent bonds between metals and organic atoms under certain conditions.
        
                The algorithm works as follows:
        
                - Disconnect N, O, F from any metal.
                - Disconnect other non-metals from transition metals + Al (but not Hg, Ga, Ge, In, Sn, As, Tl, Pb, Bi, Po).
                - For every bond broken, adjust the charges of the begin and end atoms accordingly.
        
                :param mol: The input molecule.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :return: The molecule with metals disconnected.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
log: logging.Logger  # value = <Logger rdkit.Chem.MolStandardize.metal (WARNING)>
