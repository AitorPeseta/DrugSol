"""

molvs.tautomer
~~~~~~~~~~~~~~

This module contains tools for enumerating tautomers and determining a canonical tautomer.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
from __future__ import annotations
from _warnings import warn
import copy as copy
import logging as logging
from rdkit import Chem
from rdkit.Chem.MolStandardize.utils import memoized_property
from rdkit.Chem.MolStandardize.utils import pairwise
from rdkit.Chem.rdchem import BondDir
from rdkit.Chem.rdchem import BondStereo
from rdkit.Chem.rdchem import BondType
import typing
__all__ = ['BondDir', 'BondStereo', 'BondType', 'Chem', 'MAX_TAUTOMERS', 'TAUTOMER_SCORES', 'TAUTOMER_TRANSFORMS', 'TautomerCanonicalizer', 'TautomerEnumerator', 'TautomerScore', 'TautomerTransform', 'copy', 'log', 'logging', 'memoized_property', 'pairwise', 'warn']
class TautomerCanonicalizer:
    """
    
    
        
    """
    def __call__(self, mol):
        """
        Calling a TautomerCanonicalizer instance like a function is the same as calling its canonicalize(mol) method.
        """
    def __init__(self, transforms = ..., scores = ..., max_tautomers = 1000):
        """
        
        
                :param transforms: A list of TautomerTransforms to use to enumerate tautomers.
                :param scores: A list of TautomerScores to use to choose the canonical tautomer.
                :param max_tautomers: The maximum number of tautomers to enumerate, a limit to prevent combinatorial explosion.
                
        """
    def canonicalize(self, mol):
        """
        Return a canonical tautomer by enumerating and scoring all possible tautomers.
        
                :param mol: The input molecule.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :return: The canonical tautomer.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
    @property
    def _enumerate_tautomers(self):
        ...
class TautomerEnumerator:
    """
    
    
        
    """
    def __call__(self, mol):
        """
        Calling a TautomerEnumerator instance like a function is the same as calling its enumerate(mol) method.
        """
    def __init__(self, transforms = ..., max_tautomers = 1000):
        """
        
        
                :param transforms: A list of TautomerTransforms to use to enumerate tautomers.
                :param max_tautomers: The maximum number of tautomers to enumerate (limit to prevent combinatorial explosion).
                
        """
    def enumerate(self, mol):
        """
        Enumerate all possible tautomers and return them as a list.
        
                :param mol: The input molecule.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :return: A list of all possible tautomers of the molecule.
                :rtype: list of :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
class TautomerScore:
    """
    A substructure defined by SMARTS and its score contribution to determine the canonical tautomer.
    """
    def __init__(self, name, smarts, score):
        """
        Initialize a TautomerScore with a name, SMARTS pattern and score.
        
                :param name: A name for this TautomerScore.
                :param smarts: SMARTS pattern to match a substructure.
                :param score: The score to assign for this substructure.
                
        """
    def __repr__(self):
        ...
    def __str__(self):
        ...
    @property
    def smarts(self):
        ...
class TautomerTransform:
    """
    Rules to transform one tautomer to another.
    
        Each TautomerTransform is defined by a SMARTS pattern where the transform involves moving a hydrogen from the first
        atom in the pattern to the last atom in the pattern. By default, alternating single and double bonds along the
        pattern are swapped accordingly to account for the hydrogen movement. If necessary, the transform can instead define
        custom resulting bond orders and also resulting atom charges.
        
    """
    BONDMAP: typing.ClassVar[dict]  # value = {'-': rdkit.Chem.rdchem.BondType.SINGLE, '=': rdkit.Chem.rdchem.BondType.DOUBLE, '#': rdkit.Chem.rdchem.BondType.TRIPLE, ':': rdkit.Chem.rdchem.BondType.AROMATIC}
    CHARGEMAP: typing.ClassVar[dict] = {'+': 1, '0': 0, '-': -1}
    def __init__(self, name, smarts, bonds = tuple(), charges = tuple(), radicals = tuple()):
        """
        Initialize a TautomerTransform with a name, SMARTS pattern and optional bonds and charges.
        
                The SMARTS pattern match is applied to a Kekule form of the molecule, so use explicit single and double bonds
                rather than aromatic.
        
                Specify custom bonds as a string of ``-``, ``=``, ``#``, ``:`` for single, double, triple and aromatic bonds
                respectively. Specify custom charges as ``+``, ``0``, ``-`` for +1, 0 and -1 charges respectively.
        
                :param string name: A name for this TautomerTransform.
                :param string smarts: SMARTS pattern to match for the transform.
                :param string bonds: Optional specification for the resulting bonds.
                :param string charges: Optional specification for the resulting charges on the atoms.
                
        """
    def __repr__(self):
        ...
    def __str__(self):
        ...
    @property
    def tautomer(self):
        ...
MAX_TAUTOMERS: int = 1000
TAUTOMER_SCORES: tuple  # value = (TautomerScore('benzoquinone', '[#6]1([#6]=[#6][#6]([#6]=[#6]1)=,:[N,S,O])=,:[N,S,O]', 25), TautomerScore('oxim', '[#6]=[N][OH]', 4), TautomerScore('C=O', '[#6]=,:[#8]', 2), TautomerScore('N=O', '[#7]=,:[#8]', 2), TautomerScore('P=O', '[#15]=,:[#8]', 2), TautomerScore('C=hetero', '[#6]=[!#1;!#6]', 1), TautomerScore('methyl', '[CX4H3]', 1), TautomerScore('guanidine terminal=N', '[#7][#6](=[NR0])[#7H0]', 1), TautomerScore('guanidine endocyclic=N', '[#7;R][#6;R]([N])=[#7;R]', 2), TautomerScore('aci-nitro', '[#6]=[N+]([O-])[OH]', -4))
TAUTOMER_TRANSFORMS: tuple  # value = (TautomerTransform('1,3 (thio)keto/enol f', '[CX4!H0]-[C]=[O,S,Se,Te;X1]', [], []), TautomerTransform('1,3 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[C]=[C]', [], []), TautomerTransform('1,5 (thio)keto/enol f', '[CX4,NX3;!H0]-[C]=[C][CH0]=[O,S,Se,Te;X1]', [], []), TautomerTransform('1,5 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[CH0]=[C]-[C]=[C,N]', [], []), TautomerTransform('aliphatic imine f', '[CX4!H0]-[C]=[NX2]', [], []), TautomerTransform('aliphatic imine r', '[NX3!H0]-[C]=[CX3]', [], []), TautomerTransform('special imine f', '[N!H0]-[C]=[CX3R0]', [], []), TautomerTransform('special imine r', '[CX4!H0]-[c]=[n]', [], []), TautomerTransform('1,3 aromatic heteroatom H shift f', '[#7!H0]-[#6R1]=[O,#7X2]', [], []), TautomerTransform('1,3 aromatic heteroatom H shift r', '[O,#7;!H0]-[#6R1]=[#7X2]', [], []), TautomerTransform('1,3 heteroatom H shift', '[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]', [], []), TautomerTransform('1,5 aromatic heteroatom H shift', '[#7,#16,#8;!H0]-[#6,#7]=[#6]-[#6,#7]=[#7,#16,#8;H0]', [], []), TautomerTransform('1,5 aromatic heteroatom H shift f', '[#7,#16,#8,Se,Te;!H0]-[#6,nX2]=[#6,nX2]-[#6,#7X2]=[#7X2,S,O,Se,Te]', [], []), TautomerTransform('1,5 aromatic heteroatom H shift r', '[#7,S,O,Se,Te;!H0]-[#6,#7X2]=[#6,nX2]-[#6,nX2]=[#7,#16,#8,Se,Te]', [], []), TautomerTransform('1,7 aromatic heteroatom H shift f', '[#7,#8,#16,Se,Te;!H0]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#6]-[#6,#7X2]=[#7X2,S,O,Se,Te,CX3]', [], []), TautomerTransform('1,7 aromatic heteroatom H shift r', '[#7,S,O,Se,Te,CX4;!H0]-[#6,#7X2]=[#6]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[NX2,S,O,Se,Te]', [], []), TautomerTransform('1,9 aromatic heteroatom H shift f', '[#7,O;!H0]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#7,O]', [], []), TautomerTransform('1,11 aromatic heteroatom H shift f', '[#7,O;!H0]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#7X2,O]', [], []), TautomerTransform('furanone f', '[O,S,N;!H0]-[#6r5]=[#6X3r5;$([#6]([#6r5])=[#6r5])]', [], []), TautomerTransform('furanone r', '[#6r5!H0;$([#6]([#6r5])[#6r5])]-[#6r5]=[O,S,N]', [], []), TautomerTransform('keten/ynol f', '[C!H0]=[C]=[O,S,Se,Te;X1]', [rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.SINGLE], []), TautomerTransform('keten/ynol r', '[O,S,Se,Te;!H0X2]-[C]#[C]', [rdkit.Chem.rdchem.BondType.DOUBLE, rdkit.Chem.rdchem.BondType.DOUBLE], []), TautomerTransform('ionic nitro/aci-nitro f', '[C!H0]-[N+;$([N][O-])]=[O]', [], []), TautomerTransform('ionic nitro/aci-nitro r', '[O!H0]-[N+;$([N][O-])]=[C]', [], []), TautomerTransform('oxim/nitroso f', '[O!H0]-[N]=[C]', [], []), TautomerTransform('oxim/nitroso r', '[C!H0]-[N]=[O]', [], []), TautomerTransform('oxim/nitroso via phenol f', '[O!H0]-[N]=[C]-[C]=[C]-[C]=[OH0]', [], []), TautomerTransform('oxim/nitroso via phenol r', '[O!H0]-[c]=[c]-[c]=[c]-[N]=[OH0]', [], []), TautomerTransform('cyano/iso-cyanic acid f', '[O!H0]-[C]#[N]', [rdkit.Chem.rdchem.BondType.DOUBLE, rdkit.Chem.rdchem.BondType.DOUBLE], []), TautomerTransform('cyano/iso-cyanic acid r', '[N!H0]=[C]=[O]', [rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.SINGLE], []), TautomerTransform('isocyanide f', '[C-0!H0]#[N+0]', [rdkit.Chem.rdchem.BondType.TRIPLE], [-1, 1]), TautomerTransform('isocyanide r', '[N+!H0]#[C-]', [rdkit.Chem.rdchem.BondType.TRIPLE], [-1, 1]), TautomerTransform('phosphonic acid f', '[OH]-[PH0]', [rdkit.Chem.rdchem.BondType.DOUBLE], []), TautomerTransform('phosphonic acid r', '[PH]=[O]', [rdkit.Chem.rdchem.BondType.SINGLE], []))
log: logging.Logger  # value = <Logger rdkit.Chem.MolStandardize.tautomer (WARNING)>
