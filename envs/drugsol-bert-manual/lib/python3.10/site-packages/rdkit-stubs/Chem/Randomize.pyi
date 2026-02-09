from __future__ import annotations
from rdkit import Chem
from rdkit import RDRandom as random
__all__ = ['CheckCanonicalization', 'Chem', 'RandomizeMol', 'RandomizeMolBlock', 'random']
def CheckCanonicalization(mol, nReps = 10):
    ...
def RandomizeMol(mol):
    ...
def RandomizeMolBlock(molB):
    ...
