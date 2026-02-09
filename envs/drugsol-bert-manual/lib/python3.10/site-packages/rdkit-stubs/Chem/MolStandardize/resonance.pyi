"""

molvs.resonance
~~~~~~~~~~~~~~~

Resonance (mesomeric) transformations.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
from __future__ import annotations
from _warnings import warn
import logging as logging
from rdkit import Chem
__all__ = ['Chem', 'MAX_STRUCTURES', 'ResonanceEnumerator', 'enumerate_resonance_smiles', 'log', 'logging', 'warn']
class ResonanceEnumerator:
    """
    Simple wrapper around RDKit ResonanceMolSupplier.
    
        
    """
    def __call__(self, mol):
        """
        Calling a ResonanceEnumerator instance like a function is the same as calling its enumerate(mol) method.
        """
    def __init__(self, kekule_all = False, allow_incomplete_octets = False, unconstrained_cations = False, unconstrained_anions = False, allow_charge_separation = False, max_structures = 1000):
        """
        
        
                :param bool allow_incomplete_octets: include resonance structures whose octets are less complete than the most octet-complete structure.
                :param bool allow_charge_separation: include resonance structures featuring charge separation also when uncharged resonance structures exist.
                :param bool kekule_all: enumerate all possible degenerate Kekule resonance structures (the default is to include just one).
                :param bool unconstrained_cations: if False positively charged atoms left and right of N with an incomplete octet are acceptable only if the conjugated group has a positive total formal charge.
                :param bool unconstrained_anions: if False, negatively charged atoms left of N are acceptable only if the conjugated group has a negative total formal charge.
                :param int max_structures: Maximum number of resonance forms.
                
        """
    def enumerate(self, mol):
        """
        Enumerate all possible resonance forms and return them as a list.
        
                :param mol: The input molecule.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :return: A list of all possible resonance forms of the molecule.
                :rtype: list of :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
def enumerate_resonance_smiles(smiles):
    """
    Return a set of resonance forms as SMILES strings, given a SMILES string.
    
        :param smiles: A SMILES string.
        :returns: A set containing SMILES strings for every possible resonance form.
        :rtype: set of strings.
        
    """
MAX_STRUCTURES: int = 1000
log: logging.Logger  # value = <Logger rdkit.Chem.MolStandardize.resonance (WARNING)>
