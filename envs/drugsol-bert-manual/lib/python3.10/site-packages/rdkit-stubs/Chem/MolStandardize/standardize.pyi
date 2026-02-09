"""

molvs.standardize
~~~~~~~~~~~~~~~~~

This module contains the main :class:`~molvs.standardize.Standardizer` class that can be used to perform all
standardization tasks, as well as convenience functions like :func:`~molvs.standardize.standardize_smiles` for common
standardization tasks.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
from __future__ import annotations
from _warnings import warn
import copy as copy
import logging as logging
from rdkit import Chem
from rdkit.Chem.MolStandardize.charge import Reionizer
from rdkit.Chem.MolStandardize.charge import Uncharger
from rdkit.Chem.MolStandardize.fragment import FragmentRemover
from rdkit.Chem.MolStandardize.fragment import LargestFragmentChooser
from rdkit.Chem.MolStandardize.metal import MetalDisconnector
from rdkit.Chem.MolStandardize.normalize import Normalizer
from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer
from rdkit.Chem.MolStandardize.tautomer import TautomerEnumerator
from rdkit.Chem.MolStandardize.utils import memoized_property
__all__ = ['ACID_BASE_PAIRS', 'CHARGE_CORRECTIONS', 'Chem', 'FragmentRemover', 'LargestFragmentChooser', 'MAX_RESTARTS', 'MAX_TAUTOMERS', 'MetalDisconnector', 'NORMALIZATIONS', 'Normalizer', 'PREFER_ORGANIC', 'Reionizer', 'Standardizer', 'TAUTOMER_SCORES', 'TAUTOMER_TRANSFORMS', 'TautomerCanonicalizer', 'TautomerEnumerator', 'Uncharger', 'canonicalize_tautomer_smiles', 'copy', 'enumerate_tautomers_smiles', 'log', 'logging', 'memoized_property', 'standardize_smiles', 'warn']
class Standardizer:
    """
    The main class for performing standardization of molecules and deriving parent molecules.
    
        The primary usage is via the :meth:`~molvs.standardize.Standardizer.standardize` method::
    
            s = Standardizer()
            mol1 = Chem.MolFromSmiles('C1=CC=CC=C1')
            mol2 = s.standardize(mol1)
    
        There are separate methods to derive fragment, charge, tautomer, isotope and stereo parent molecules.
    
        
    """
    def __call__(self, mol):
        """
        Calling a Standardizer instance like a function is the same as calling its
                :meth:`~molvs.standardize.Standardizer.standardize` method.
        """
    def __init__(self, normalizations = ..., acid_base_pairs = ..., charge_corrections = ..., tautomer_transforms = ..., tautomer_scores = ..., max_restarts = 200, max_tautomers = 1000, prefer_organic = False):
        """
        Initialize a Standardizer with optional custom parameters.
        
                :param normalizations: A list of Normalizations to apply (default: :data:`~molvs.normalize.NORMALIZATIONS`).
                :param acid_base_pairs: A list of AcidBasePairs for competitive reionization (default:
                                        :data:`~molvs.charge.ACID_BASE_PAIRS`).
                :param charge_corrections: A list of ChargeCorrections to apply (default:
                                        :data:`~molvs.charge.CHARGE_CORRECTIONS`).
                :param tautomer_transforms: A list of TautomerTransforms to apply (default:
                                            :data:`~molvs.tautomer.TAUTOMER_TRANSFORMS`).
                :param tautomer_scores: A list of TautomerScores used to determine canonical tautomer (default:
                                        :data:`~molvs.tautomer.TAUTOMER_SCORES`).
                :param max_restarts: The maximum number of times to attempt to apply the series of normalizations (default 200).
                :param max_tautomers: The maximum number of tautomers to enumerate (default 1000).
                :param prefer_organic: Whether to prioritize organic fragments when choosing fragment parent (default False).
                
        """
    def charge_parent(self, mol, skip_standardize = False):
        """
        Return the charge parent of a given molecule.
        
                The charge parent is the uncharged version of the fragment parent.
        
                :param mol: The input molecule.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :param bool skip_standardize: Set to True if mol has already been standardized.
                :returns: The charge parent molecule.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
    def fragment_parent(self, mol, skip_standardize = False):
        """
        Return the fragment parent of a given molecule.
        
                The fragment parent is the largest organic covalent unit in the molecule.
        
                :param mol: The input molecule.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :param bool skip_standardize: Set to True if mol has already been standardized.
                :returns: The fragment parent molecule.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
    def isotope_parent(self, mol, skip_standardize = False):
        """
        Return the isotope parent of a given molecule.
        
                The isotope parent has all atoms replaced with the most abundant isotope for that element.
        
                :param mol: The input molecule.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :param bool skip_standardize: Set to True if mol has already been standardized.
                :returns: The isotope parent molecule.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
    def standardize(self, mol):
        """
        Return a standardized version the given molecule.
        
                The standardization process consists of the following stages: RDKit
                :rdkit:`RemoveHs <Chem.rdmolops-module.html#RemoveHs>`, RDKit
                :rdkit:`SanitizeMol <Chem.rdmolops-module.html#SanitizeMol>`, :class:`~molvs.metal.MetalDisconnector`,
                :class:`~molvs.normalize.Normalizer`, :class:`~molvs.charge.Reionizer`, RDKit
                :rdkit:`AssignStereochemistry <Chem.rdmolops-module.html#AssignStereochemistry>`.
        
                :param mol: The molecule to standardize.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :returns: The standardized molecule.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
    def standardize_with_parents(self, mol):
        """
        """
    def stereo_parent(self, mol, skip_standardize = False):
        """
        Return the stereo parent of a given molecule.
        
                The stereo parent has all stereochemistry information removed from tetrahedral centers and double bonds.
        
                :param mol: The input molecule.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :param bool skip_standardize: Set to True if mol has already been standardized.
                :returns: The stereo parent molecule.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
    def super_parent(self, mol, skip_standardize = False):
        """
        Return the super parent of a given molecule.
        
                THe super parent is fragment, charge, isotope, stereochemistry and tautomer insensitive. From the input
                molecule, the largest fragment is taken. This is uncharged and then isotope and stereochemistry information is
                discarded. Finally, the canonical tautomer is determined and returned.
        
                :param mol: The input molecule.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :param bool skip_standardize: Set to True if mol has already been standardized.
                :returns: The super parent molecule.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
    def tautomer_parent(self, mol, skip_standardize = False):
        """
        Return the tautomer parent of a given molecule.
        
                :param mol: The input molecule.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :param bool skip_standardize: Set to True if mol has already been standardized.
                :returns: The tautomer parent molecule.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
    @property
    def canonicalize_tautomer(self):
        """
        
                :returns: A callable :class:`~molvs.tautomer.TautomerCanonicalizer` instance.
                
        """
    @property
    def disconnect_metals(self):
        """
        
                :returns: A callable :class:`~molvs.metal.MetalDisconnector` instance.
                
        """
    @property
    def enumerate_tautomers(self):
        """
        
                :returns: A callable :class:`~molvs.tautomer.TautomerEnumerator` instance.
                
        """
    @property
    def largest_fragment(self):
        """
        
                :returns: A callable :class:`~molvs.fragment.LargestFragmentChooser` instance.
                
        """
    @property
    def normalize(self):
        """
        
                :returns: A callable :class:`~molvs.normalize.Normalizer` instance.
                
        """
    @property
    def reionize(self):
        """
        
                :returns: A callable :class:`~molvs.charge.Reionizer` instance.
                
        """
    @property
    def remove_fragments(self):
        """
        
                :returns: A callable :class:`~molvs.fragment.FragmentRemover` instance.
                
        """
    @property
    def uncharge(self):
        """
        
                :returns: A callable :class:`~molvs.charge.Uncharger` instance.
                
        """
def canonicalize_tautomer_smiles(smiles):
    """
    Return a standardized canonical tautomer SMILES string given a SMILES string.
    
        Note: This is a convenience function for quickly standardizing and finding the canonical tautomer for a single
        SMILES string. It is more efficient to use the :class:`~molvs.standardize.Standardizer` class directly when working
        with many molecules or when custom options are needed.
    
        :param string smiles: The SMILES for the molecule.
        :returns: The SMILES for the standardize canonical tautomer.
        :rtype: string.
        
    """
def enumerate_tautomers_smiles(smiles):
    """
    Return a set of tautomers as SMILES strings, given a SMILES string.
    
        :param smiles: A SMILES string.
        :returns: A set containing SMILES strings for every possible tautomer.
        :rtype: set of strings.
        
    """
def standardize_smiles(smiles):
    """
    Return a standardized canonical SMILES string given a SMILES string.
    
        Note: This is a convenience function for quickly standardizing a single SMILES string. It is more efficient to use
        the :class:`~molvs.standardize.Standardizer` class directly when working with many molecules or when custom options
        are needed.
    
        :param string smiles: The SMILES for the molecule.
        :returns: The SMILES for the standardized molecule.
        :rtype: string.
        
    """
ACID_BASE_PAIRS: tuple  # value = (AcidBasePair('-OSO3H', 'OS(=O)(=O)[OH]', 'OS(=O)(=O)[O-]'), AcidBasePair('-SO3H', '[!O]S(=O)(=O)[OH]', '[!O]S(=O)(=O)[O-]'), AcidBasePair('-OSO2H', 'O[SD3](=O)[OH]', 'O[SD3](=O)[O-]'), AcidBasePair('-SO2H', '[!O][SD3](=O)[OH]', '[!O][SD3](=O)[O-]'), AcidBasePair('-OPO3H2', 'OP(=O)([OH])[OH]', 'OP(=O)([OH])[O-]'), AcidBasePair('-PO3H2', '[!O]P(=O)([OH])[OH]', '[!O]P(=O)([OH])[O-]'), AcidBasePair('-CO2H', 'C(=O)[OH]', 'C(=O)[O-]'), AcidBasePair('thiophenol', 'c[SH]', 'c[S-]'), AcidBasePair('(-OPO3H)-', 'OP(=O)([O-])[OH]', 'OP(=O)([O-])[O-]'), AcidBasePair('(-PO3H)-', '[!O]P(=O)([O-])[OH]', '[!O]P(=O)([O-])[O-]'), AcidBasePair('phthalimide', 'O=C2c1ccccc1C(=O)[NH]2', 'O=C2c1ccccc1C(=O)[N-]2'), AcidBasePair('CO3H (peracetyl)', 'C(=O)O[OH]', 'C(=O)O[O-]'), AcidBasePair('alpha-carbon-hydrogen-nitro group', 'O=N(O)[CH]', 'O=N(O)[C-]'), AcidBasePair('-SO2NH2', 'S(=O)(=O)[NH2]', 'S(=O)(=O)[NH-]'), AcidBasePair('-OBO2H2', 'OB([OH])[OH]', 'OB([OH])[O-]'), AcidBasePair('-BO2H2', '[!O]B([OH])[OH]', '[!O]B([OH])[O-]'), AcidBasePair('phenol', 'c[OH]', 'c[O-]'), AcidBasePair('SH (aliphatic)', 'C[SH]', 'C[S-]'), AcidBasePair('(-OBO2H)-', 'OB([O-])[OH]', 'OB([O-])[O-]'), AcidBasePair('(-BO2H)-', '[!O]B([O-])[OH]', '[!O]B([O-])[O-]'), AcidBasePair('cyclopentadiene', 'C1=CC=C[CH2]1', 'c1ccc[cH-]1'), AcidBasePair('-CONH2', 'C(=O)[NH2]', 'C(=O)[NH-]'), AcidBasePair('imidazole', 'c1cnc[nH]1', 'c1cnc[n-]1'), AcidBasePair('-OH (aliphatic alcohol)', '[CX4][OH]', '[CX4][O-]'), AcidBasePair('alpha-carbon-hydrogen-keto group', 'O=C([!O])[C!H0+0]', 'O=C([!O])[C-]'), AcidBasePair('alpha-carbon-hydrogen-acetyl ester group', 'OC(=O)[C!H0+0]', 'OC(=O)[C-]'), AcidBasePair('sp carbon hydrogen', 'C#[CH]', 'C#[C-]'), AcidBasePair('alpha-carbon-hydrogen-sulfone group', 'CS(=O)(=O)[C!H0+0]', 'CS(=O)(=O)[C-]'), AcidBasePair('alpha-carbon-hydrogen-sulfoxide group', 'C[SD3](=O)[C!H0+0]', 'C[SD3](=O)[C-]'), AcidBasePair('-NH2', '[CX4][NH2]', '[CX4][NH-]'), AcidBasePair('benzyl hydrogen', 'c[CX4H2]', 'c[CX3H-]'), AcidBasePair('sp2-carbon hydrogen', '[CX3]=[CX3!H0+0]', '[CX3]=[CX2-]'), AcidBasePair('sp3-carbon hydrogen', '[CX4!H0+0]', '[CX3-]'))
CHARGE_CORRECTIONS: tuple  # value = (ChargeCorrection('[Li,Na,K]', '[Li,Na,K;X0+0]', 1), ChargeCorrection('[Mg,Ca]', '[Mg,Ca;X0+0]', 2), ChargeCorrection('[Cl]', '[Cl;X0+0]', -1))
MAX_RESTARTS: int = 200
MAX_TAUTOMERS: int = 1000
NORMALIZATIONS: tuple  # value = (Normalization('Nitro to N+(O-)=O', '[N,P,As,Sb;X3:1](=[O,S,Se,Te:2])=[O,S,Se,Te:3]>>[*+1:1]([*-1:2])=[*:3]'), Normalization('Sulfone to S(=O)(=O)', '[S+2:1]([O-:2])([O-:3])>>[S+0:1](=[O-0:2])(=[O-0:3])'), Normalization('Pyridine oxide to n+O-', '[n:1]=[O:2]>>[n+:1][O-:2]'), Normalization('Azide to N=N+=N-', '[*,H:1][N:2]=[N:3]#[N:4]>>[*,H:1][N:2]=[N+:3]=[N-:4]'), Normalization('Diazo/azo to =N+=N-', '[*:1]=[N:2]#[N:3]>>[*:1]=[N+:2]=[N-:3]'), Normalization('Sulfoxide to -S+(O-)-', '[!O:1][S+0;X3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]'), Normalization('Phosphate to P(O-)=O', '[O,S,Se,Te;-1:1][P+;D4:2][O,S,Se,Te;-1:3]>>[*+0:1]=[P+0;D5:2][*-1:3]'), Normalization('C/S+N to C/S=N+', '[C,S;X3+1:1]([NX3:2])[NX3!H0:3]>>[*+0:1]([N:2])=[N+:3]'), Normalization('P+N to P=N+', '[P;X4+1:1]([NX3:2])[NX3!H0:3]>>[*+0:1]([N:2])=[N+:3]'), Normalization('Normalize hydrazine-diazonium', '[CX4:1][NX3H:2]-[NX3H:3][CX4:4][NX2+:5]#[NX1:6]>>[CX4:1][NH0:2]=[NH+:3][C:4][N+0:5]=[NH:6]'), Normalization('Recombine 1,3-separated charges', '[N,P,As,Sb,O,S,Se,Te;-1:1]-[A+0:2]=[N,P,As,Sb,O,S,Se,Te;+1:3]>>[*-0:1]=[*:2]-[*+0:3]'), Normalization('Recombine 1,3-separated charges', '[n,o,p,s;-1:1]:[a:2]=[N,O,P,S;+1:3]>>[*-0:1]:[*:2]-[*+0:3]'), Normalization('Recombine 1,3-separated charges', '[N,O,P,S;-1:1]-[a:2]:[n,o,p,s;+1:3]>>[*-0:1]=[*:2]:[*+0:3]'), Normalization('Recombine 1,5-separated charges', '[N,P,As,Sb,O,S,Se,Te;-1:1]-[A+0:2]=[A:3]-[A:4]=[N,P,As,Sb,O,S,Se,Te;+1:5]>>[*-0:1]=[*:2]-[*:3]=[*:4]-[*+0:5]'), Normalization('Recombine 1,5-separated charges', '[n,o,p,s;-1:1]:[a:2]:[a:3]:[c:4]=[N,O,P,S;+1:5]>>[*-0:1]:[*:2]:[*:3]:[c:4]-[*+0:5]'), Normalization('Recombine 1,5-separated charges', '[N,O,P,S;-1:1]-[c:2]:[a:3]:[a:4]:[n,o,p,s;+1:5]>>[*-0:1]=[c:2]:[*:3]:[*:4]:[*+0:5]'), Normalization('Normalize 1,3 conjugated cation', '[N,O;+0!H0:1]-[A:2]=[N!$(*[O-]),O;+1H0:3]>>[*+1:1]=[*:2]-[*+0:3]'), Normalization('Normalize 1,3 conjugated cation', '[n;+0!H0:1]:[c:2]=[N!$(*[O-]),O;+1H0:3]>>[*+1:1]:[*:2]-[*+0:3]'), Normalization('Normalize 1,5 conjugated cation', '[N,O;+0!H0:1]-[A:2]=[A:3]-[A:4]=[N!$(*[O-]),O;+1H0:5]>>[*+1:1]=[*:2]-[*:3]=[*:4]-[*+0:5]'), Normalization('Normalize 1,5 conjugated cation', '[n;+0!H0:1]:[a:2]:[a:3]:[c:4]=[N!$(*[O-]),O;+1H0:5]>>[n+1:1]:[*:2]:[*:3]:[*:4]-[*+0:5]'), Normalization('Charge normalization', '[F,Cl,Br,I,At;-1:1]=[O:2]>>[*-0:1][O-:2]'), Normalization('Charge recombination', '[N,P,As,Sb;-1:1]=[C+;v3:2]>>[*+0:1]#[C+0:2]'))
PREFER_ORGANIC: bool = False
TAUTOMER_SCORES: tuple  # value = (TautomerScore('benzoquinone', '[#6]1([#6]=[#6][#6]([#6]=[#6]1)=,:[N,S,O])=,:[N,S,O]', 25), TautomerScore('oxim', '[#6]=[N][OH]', 4), TautomerScore('C=O', '[#6]=,:[#8]', 2), TautomerScore('N=O', '[#7]=,:[#8]', 2), TautomerScore('P=O', '[#15]=,:[#8]', 2), TautomerScore('C=hetero', '[#6]=[!#1;!#6]', 1), TautomerScore('methyl', '[CX4H3]', 1), TautomerScore('guanidine terminal=N', '[#7][#6](=[NR0])[#7H0]', 1), TautomerScore('guanidine endocyclic=N', '[#7;R][#6;R]([N])=[#7;R]', 2), TautomerScore('aci-nitro', '[#6]=[N+]([O-])[OH]', -4))
TAUTOMER_TRANSFORMS: tuple  # value = (TautomerTransform('1,3 (thio)keto/enol f', '[CX4!H0]-[C]=[O,S,Se,Te;X1]', [], []), TautomerTransform('1,3 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[C]=[C]', [], []), TautomerTransform('1,5 (thio)keto/enol f', '[CX4,NX3;!H0]-[C]=[C][CH0]=[O,S,Se,Te;X1]', [], []), TautomerTransform('1,5 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[CH0]=[C]-[C]=[C,N]', [], []), TautomerTransform('aliphatic imine f', '[CX4!H0]-[C]=[NX2]', [], []), TautomerTransform('aliphatic imine r', '[NX3!H0]-[C]=[CX3]', [], []), TautomerTransform('special imine f', '[N!H0]-[C]=[CX3R0]', [], []), TautomerTransform('special imine r', '[CX4!H0]-[c]=[n]', [], []), TautomerTransform('1,3 aromatic heteroatom H shift f', '[#7!H0]-[#6R1]=[O,#7X2]', [], []), TautomerTransform('1,3 aromatic heteroatom H shift r', '[O,#7;!H0]-[#6R1]=[#7X2]', [], []), TautomerTransform('1,3 heteroatom H shift', '[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]', [], []), TautomerTransform('1,5 aromatic heteroatom H shift', '[#7,#16,#8;!H0]-[#6,#7]=[#6]-[#6,#7]=[#7,#16,#8;H0]', [], []), TautomerTransform('1,5 aromatic heteroatom H shift f', '[#7,#16,#8,Se,Te;!H0]-[#6,nX2]=[#6,nX2]-[#6,#7X2]=[#7X2,S,O,Se,Te]', [], []), TautomerTransform('1,5 aromatic heteroatom H shift r', '[#7,S,O,Se,Te;!H0]-[#6,#7X2]=[#6,nX2]-[#6,nX2]=[#7,#16,#8,Se,Te]', [], []), TautomerTransform('1,7 aromatic heteroatom H shift f', '[#7,#8,#16,Se,Te;!H0]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#6]-[#6,#7X2]=[#7X2,S,O,Se,Te,CX3]', [], []), TautomerTransform('1,7 aromatic heteroatom H shift r', '[#7,S,O,Se,Te,CX4;!H0]-[#6,#7X2]=[#6]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[NX2,S,O,Se,Te]', [], []), TautomerTransform('1,9 aromatic heteroatom H shift f', '[#7,O;!H0]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#6,#7X2]-[#6,#7X2]=[#7,O]', [], []), TautomerTransform('1,11 aromatic heteroatom H shift f', '[#7,O;!H0]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#6,nX2]-[#6,nX2]=[#7X2,O]', [], []), TautomerTransform('furanone f', '[O,S,N;!H0]-[#6r5]=[#6X3r5;$([#6]([#6r5])=[#6r5])]', [], []), TautomerTransform('furanone r', '[#6r5!H0;$([#6]([#6r5])[#6r5])]-[#6r5]=[O,S,N]', [], []), TautomerTransform('keten/ynol f', '[C!H0]=[C]=[O,S,Se,Te;X1]', [rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.SINGLE], []), TautomerTransform('keten/ynol r', '[O,S,Se,Te;!H0X2]-[C]#[C]', [rdkit.Chem.rdchem.BondType.DOUBLE, rdkit.Chem.rdchem.BondType.DOUBLE], []), TautomerTransform('ionic nitro/aci-nitro f', '[C!H0]-[N+;$([N][O-])]=[O]', [], []), TautomerTransform('ionic nitro/aci-nitro r', '[O!H0]-[N+;$([N][O-])]=[C]', [], []), TautomerTransform('oxim/nitroso f', '[O!H0]-[N]=[C]', [], []), TautomerTransform('oxim/nitroso r', '[C!H0]-[N]=[O]', [], []), TautomerTransform('oxim/nitroso via phenol f', '[O!H0]-[N]=[C]-[C]=[C]-[C]=[OH0]', [], []), TautomerTransform('oxim/nitroso via phenol r', '[O!H0]-[c]=[c]-[c]=[c]-[N]=[OH0]', [], []), TautomerTransform('cyano/iso-cyanic acid f', '[O!H0]-[C]#[N]', [rdkit.Chem.rdchem.BondType.DOUBLE, rdkit.Chem.rdchem.BondType.DOUBLE], []), TautomerTransform('cyano/iso-cyanic acid r', '[N!H0]=[C]=[O]', [rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.SINGLE], []), TautomerTransform('isocyanide f', '[C-0!H0]#[N+0]', [rdkit.Chem.rdchem.BondType.TRIPLE], [-1, 1]), TautomerTransform('isocyanide r', '[N+!H0]#[C-]', [rdkit.Chem.rdchem.BondType.TRIPLE], [-1, 1]), TautomerTransform('phosphonic acid f', '[OH]-[PH0]', [rdkit.Chem.rdchem.BondType.DOUBLE], []), TautomerTransform('phosphonic acid r', '[PH]=[O]', [rdkit.Chem.rdchem.BondType.SINGLE], []))
__warningregistry__: dict = {'version': 4}
log: logging.Logger  # value = <Logger rdkit.Chem.MolStandardize.standardize (WARNING)>
