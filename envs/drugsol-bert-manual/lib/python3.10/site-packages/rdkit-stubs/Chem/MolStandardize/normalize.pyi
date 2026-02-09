"""

molvs.normalize
~~~~~~~~~~~~~~~

This module contains tools for normalizing molecules using reaction SMARTS patterns.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
from __future__ import annotations
from _warnings import warn
import logging as logging
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize.utils import memoized_property
__all__ = ['AllChem', 'Chem', 'MAX_RESTARTS', 'NORMALIZATIONS', 'Normalization', 'Normalizer', 'log', 'logging', 'memoized_property', 'warn']
class Normalization:
    """
    A normalization transform defined by reaction SMARTS.
    """
    def __init__(self, name, transform):
        """
        
                :param string name: A name for this Normalization
                :param string transform: Reaction SMARTS to define the transformation.
                
        """
    def __repr__(self):
        ...
    def __str__(self):
        ...
    @property
    def transform(self):
        ...
class Normalizer:
    """
    A class for applying Normalization transforms.
    
        This class is typically used to apply a series of Normalization transforms to correct functional groups and
        recombine charges. Each transform is repeatedly applied until no further changes occur.
        
    """
    def __call__(self, mol):
        """
        Calling a Normalizer instance like a function is the same as calling its normalize(mol) method.
        """
    def __init__(self, normalizations = ..., max_restarts = 200):
        """
        Initialize a Normalizer with an optional custom list of :class:`~molvs.normalize.Normalization` transforms.
        
                :param normalizations: A list of  :class:`~molvs.normalize.Normalization` transforms to apply.
                :param int max_restarts: The maximum number of times to attempt to apply the series of normalizations (default
                                         200).
                
        """
    def _apply_transform(self, mol, rule):
        """
        Repeatedly apply normalization transform to molecule until no changes occur.
        
                It is possible for multiple products to be produced when a rule is applied. The rule is applied repeatedly to
                each of the products, until no further changes occur or after 20 attempts. If there are multiple unique products
                after the final application, the first product (sorted alphabetically by SMILES) is chosen.
                
        """
    def _normalize_fragment(self, mol):
        ...
    def normalize(self, mol):
        """
        Apply a series of Normalization transforms to correct functional groups and recombine charges.
        
                A series of transforms are applied to the molecule. For each Normalization, the transform is applied repeatedly
                until no further changes occur. If any changes occurred, we go back and start from the first Normalization
                again, in case the changes mean an earlier transform is now applicable. The molecule is returned once the entire
                series of Normalizations cause no further changes or if max_restarts (default 200) is reached.
        
                :param mol: The molecule to normalize.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :return: The normalized fragment.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
MAX_RESTARTS: int = 200
NORMALIZATIONS: tuple  # value = (Normalization('Nitro to N+(O-)=O', '[N,P,As,Sb;X3:1](=[O,S,Se,Te:2])=[O,S,Se,Te:3]>>[*+1:1]([*-1:2])=[*:3]'), Normalization('Sulfone to S(=O)(=O)', '[S+2:1]([O-:2])([O-:3])>>[S+0:1](=[O-0:2])(=[O-0:3])'), Normalization('Pyridine oxide to n+O-', '[n:1]=[O:2]>>[n+:1][O-:2]'), Normalization('Azide to N=N+=N-', '[*,H:1][N:2]=[N:3]#[N:4]>>[*,H:1][N:2]=[N+:3]=[N-:4]'), Normalization('Diazo/azo to =N+=N-', '[*:1]=[N:2]#[N:3]>>[*:1]=[N+:2]=[N-:3]'), Normalization('Sulfoxide to -S+(O-)-', '[!O:1][S+0;X3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]'), Normalization('Phosphate to P(O-)=O', '[O,S,Se,Te;-1:1][P+;D4:2][O,S,Se,Te;-1:3]>>[*+0:1]=[P+0;D5:2][*-1:3]'), Normalization('C/S+N to C/S=N+', '[C,S;X3+1:1]([NX3:2])[NX3!H0:3]>>[*+0:1]([N:2])=[N+:3]'), Normalization('P+N to P=N+', '[P;X4+1:1]([NX3:2])[NX3!H0:3]>>[*+0:1]([N:2])=[N+:3]'), Normalization('Normalize hydrazine-diazonium', '[CX4:1][NX3H:2]-[NX3H:3][CX4:4][NX2+:5]#[NX1:6]>>[CX4:1][NH0:2]=[NH+:3][C:4][N+0:5]=[NH:6]'), Normalization('Recombine 1,3-separated charges', '[N,P,As,Sb,O,S,Se,Te;-1:1]-[A+0:2]=[N,P,As,Sb,O,S,Se,Te;+1:3]>>[*-0:1]=[*:2]-[*+0:3]'), Normalization('Recombine 1,3-separated charges', '[n,o,p,s;-1:1]:[a:2]=[N,O,P,S;+1:3]>>[*-0:1]:[*:2]-[*+0:3]'), Normalization('Recombine 1,3-separated charges', '[N,O,P,S;-1:1]-[a:2]:[n,o,p,s;+1:3]>>[*-0:1]=[*:2]:[*+0:3]'), Normalization('Recombine 1,5-separated charges', '[N,P,As,Sb,O,S,Se,Te;-1:1]-[A+0:2]=[A:3]-[A:4]=[N,P,As,Sb,O,S,Se,Te;+1:5]>>[*-0:1]=[*:2]-[*:3]=[*:4]-[*+0:5]'), Normalization('Recombine 1,5-separated charges', '[n,o,p,s;-1:1]:[a:2]:[a:3]:[c:4]=[N,O,P,S;+1:5]>>[*-0:1]:[*:2]:[*:3]:[c:4]-[*+0:5]'), Normalization('Recombine 1,5-separated charges', '[N,O,P,S;-1:1]-[c:2]:[a:3]:[a:4]:[n,o,p,s;+1:5]>>[*-0:1]=[c:2]:[*:3]:[*:4]:[*+0:5]'), Normalization('Normalize 1,3 conjugated cation', '[N,O;+0!H0:1]-[A:2]=[N!$(*[O-]),O;+1H0:3]>>[*+1:1]=[*:2]-[*+0:3]'), Normalization('Normalize 1,3 conjugated cation', '[n;+0!H0:1]:[c:2]=[N!$(*[O-]),O;+1H0:3]>>[*+1:1]:[*:2]-[*+0:3]'), Normalization('Normalize 1,5 conjugated cation', '[N,O;+0!H0:1]-[A:2]=[A:3]-[A:4]=[N!$(*[O-]),O;+1H0:5]>>[*+1:1]=[*:2]-[*:3]=[*:4]-[*+0:5]'), Normalization('Normalize 1,5 conjugated cation', '[n;+0!H0:1]:[a:2]:[a:3]:[c:4]=[N!$(*[O-]),O;+1H0:5]>>[n+1:1]:[*:2]:[*:3]:[*:4]-[*+0:5]'), Normalization('Charge normalization', '[F,Cl,Br,I,At;-1:1]=[O:2]>>[*-0:1][O-:2]'), Normalization('Charge recombination', '[N,P,As,Sb;-1:1]=[C+;v3:2]>>[*+0:1]#[C+0:2]'))
log: logging.Logger  # value = <Logger rdkit.Chem.MolStandardize.normalize (WARNING)>
