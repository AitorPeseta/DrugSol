"""

molvs.charge
~~~~~~~~~~~~

This module implements tools for manipulating charges on molecules. In particular, :class:`~molvs.charge.Reionizer`,
which competitively reionizes acids such that the strongest acids ionize first, and :class:`~molvs.charge.Uncharger`,
which attempts to neutralize ionized acids and bases on a molecule.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
from __future__ import annotations
from _warnings import warn
import copy as copy
import logging as logging
from rdkit import Chem
from rdkit.Chem.MolStandardize.utils import memoized_property
__all__ = ['ACID_BASE_PAIRS', 'AcidBasePair', 'CHARGE_CORRECTIONS', 'ChargeCorrection', 'Chem', 'Reionizer', 'Uncharger', 'copy', 'log', 'logging', 'memoized_property', 'warn']
class AcidBasePair:
    """
    An acid and its conjugate base, defined by SMARTS.
    
        A strength-ordered list of AcidBasePairs can be used to ensure the strongest acids in a molecule ionize first.
        
    """
    def __init__(self, name, acid, base):
        """
        Initialize an AcidBasePair with the following parameters:
        
                :param string name: A name for this AcidBasePair.
                :param string acid: SMARTS pattern for the protonated acid.
                :param string base: SMARTS pattern for the conjugate ionized base.
                
        """
    def __repr__(self):
        ...
    def __str__(self):
        ...
    @property
    def acid(self):
        ...
    @property
    def base(self):
        ...
class ChargeCorrection:
    """
    An atom that should have a certain charge applied, defined by a SMARTS pattern.
    """
    def __init__(self, name, smarts, charge):
        """
        Initialize a ChargeCorrection with the following parameters:
        
                :param string name: A name for this ForcedAtomCharge.
                :param string smarts: SMARTS pattern to match. Charge is applied to the first atom.
                :param int charge: The charge to apply.
                
        """
    def __repr__(self):
        ...
    def __str__(self):
        ...
    @property
    def smarts(self):
        ...
class Reionizer:
    """
    A class to fix charges and reionize a molecule such that the strongest acids ionize first.
    """
    def __call__(self, mol):
        """
        Calling a Reionizer instance like a function is the same as calling its reionize(mol) method.
        """
    def __init__(self, acid_base_pairs = ..., charge_corrections = ...):
        """
        Initialize a Reionizer with the following parameter:
        
                :param acid_base_pairs: A list of :class:`AcidBasePairs <molvs.charge.AcidBasePair>` to reionize, sorted from
                                        strongest to weakest.
                :param charge_corrections: A list of :class:`ChargeCorrections <molvs.charge.ChargeCorrection>`.
                
        """
    def _strongest_protonated(self, mol):
        ...
    def _weakest_ionized(self, mol):
        ...
    def reionize(self, mol):
        """
        Enforce charges on certain atoms, then perform competitive reionization.
        
                First, charge corrections are applied to ensure, for example, that free metals are correctly ionized. Then, if
                a molecule with multiple acid groups is partially ionized, ensure the strongest acids ionize first.
        
                The algorithm works as follows:
        
                - Use SMARTS to find the strongest protonated acid and the weakest ionized acid.
                - If the ionized acid is weaker than the protonated acid, swap proton and repeat.
        
                :param mol: The molecule to reionize.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :return: The reionized molecule.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
class Uncharger:
    """
    Class for neutralizing ionized acids and bases.
    
        This class uncharges molecules by adding and/or removing hydrogens. For zwitterions, hydrogens are moved to
        eliminate charges where possible. However, in cases where there is a positive charge that is not neutralizable, an
        attempt is made to also preserve the corresponding negative charge.
    
        The method is derived from the neutralise module in `Francis Atkinson's standardiser tool
        <https://github.com/flatkinson/standardiser>`_, which is released under the Apache License v2.0.
        
    """
    def __call__(self, mol):
        """
        Calling an Uncharger instance like a function is the same as calling its uncharge(mol) method.
        """
    def __init__(self):
        ...
    def uncharge(self, mol):
        """
        Neutralize molecule by adding/removing hydrogens. Attempts to preserve zwitterions.
        
                :param mol: The molecule to uncharge.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :return: The uncharged molecule.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
ACID_BASE_PAIRS: tuple  # value = (AcidBasePair('-OSO3H', 'OS(=O)(=O)[OH]', 'OS(=O)(=O)[O-]'), AcidBasePair('-SO3H', '[!O]S(=O)(=O)[OH]', '[!O]S(=O)(=O)[O-]'), AcidBasePair('-OSO2H', 'O[SD3](=O)[OH]', 'O[SD3](=O)[O-]'), AcidBasePair('-SO2H', '[!O][SD3](=O)[OH]', '[!O][SD3](=O)[O-]'), AcidBasePair('-OPO3H2', 'OP(=O)([OH])[OH]', 'OP(=O)([OH])[O-]'), AcidBasePair('-PO3H2', '[!O]P(=O)([OH])[OH]', '[!O]P(=O)([OH])[O-]'), AcidBasePair('-CO2H', 'C(=O)[OH]', 'C(=O)[O-]'), AcidBasePair('thiophenol', 'c[SH]', 'c[S-]'), AcidBasePair('(-OPO3H)-', 'OP(=O)([O-])[OH]', 'OP(=O)([O-])[O-]'), AcidBasePair('(-PO3H)-', '[!O]P(=O)([O-])[OH]', '[!O]P(=O)([O-])[O-]'), AcidBasePair('phthalimide', 'O=C2c1ccccc1C(=O)[NH]2', 'O=C2c1ccccc1C(=O)[N-]2'), AcidBasePair('CO3H (peracetyl)', 'C(=O)O[OH]', 'C(=O)O[O-]'), AcidBasePair('alpha-carbon-hydrogen-nitro group', 'O=N(O)[CH]', 'O=N(O)[C-]'), AcidBasePair('-SO2NH2', 'S(=O)(=O)[NH2]', 'S(=O)(=O)[NH-]'), AcidBasePair('-OBO2H2', 'OB([OH])[OH]', 'OB([OH])[O-]'), AcidBasePair('-BO2H2', '[!O]B([OH])[OH]', '[!O]B([OH])[O-]'), AcidBasePair('phenol', 'c[OH]', 'c[O-]'), AcidBasePair('SH (aliphatic)', 'C[SH]', 'C[S-]'), AcidBasePair('(-OBO2H)-', 'OB([O-])[OH]', 'OB([O-])[O-]'), AcidBasePair('(-BO2H)-', '[!O]B([O-])[OH]', '[!O]B([O-])[O-]'), AcidBasePair('cyclopentadiene', 'C1=CC=C[CH2]1', 'c1ccc[cH-]1'), AcidBasePair('-CONH2', 'C(=O)[NH2]', 'C(=O)[NH-]'), AcidBasePair('imidazole', 'c1cnc[nH]1', 'c1cnc[n-]1'), AcidBasePair('-OH (aliphatic alcohol)', '[CX4][OH]', '[CX4][O-]'), AcidBasePair('alpha-carbon-hydrogen-keto group', 'O=C([!O])[C!H0+0]', 'O=C([!O])[C-]'), AcidBasePair('alpha-carbon-hydrogen-acetyl ester group', 'OC(=O)[C!H0+0]', 'OC(=O)[C-]'), AcidBasePair('sp carbon hydrogen', 'C#[CH]', 'C#[C-]'), AcidBasePair('alpha-carbon-hydrogen-sulfone group', 'CS(=O)(=O)[C!H0+0]', 'CS(=O)(=O)[C-]'), AcidBasePair('alpha-carbon-hydrogen-sulfoxide group', 'C[SD3](=O)[C!H0+0]', 'C[SD3](=O)[C-]'), AcidBasePair('-NH2', '[CX4][NH2]', '[CX4][NH-]'), AcidBasePair('benzyl hydrogen', 'c[CX4H2]', 'c[CX3H-]'), AcidBasePair('sp2-carbon hydrogen', '[CX3]=[CX3!H0+0]', '[CX3]=[CX2-]'), AcidBasePair('sp3-carbon hydrogen', '[CX4!H0+0]', '[CX3-]'))
CHARGE_CORRECTIONS: tuple  # value = (ChargeCorrection('[Li,Na,K]', '[Li,Na,K;X0+0]', 1), ChargeCorrection('[Mg,Ca]', '[Mg,Ca;X0+0]', 2), ChargeCorrection('[Cl]', '[Cl;X0+0]', -1))
log: logging.Logger  # value = <Logger rdkit.Chem.MolStandardize.charge (WARNING)>
