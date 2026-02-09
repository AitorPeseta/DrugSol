"""

molvs.fragment
~~~~~~~~~~~~~~

This module contains tools for dealing with molecules with more than one covalently bonded unit. The main classes are
:class:`~molvs.fragment.LargestFragmentChooser`, which returns the largest covalent unit in a molecule, and
:class:`~molvs.fragment.FragmentRemover`, which filters out fragments from a molecule using SMARTS patterns.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
from __future__ import annotations
from _warnings import warn
import logging as logging
from rdkit import Chem
from rdkit.Chem.MolStandardize.utils import memoized_property
from rdkit.Chem import rdMolDescriptors
__all__ = ['Chem', 'FragmentPattern', 'FragmentRemover', 'LEAVE_LAST', 'LargestFragmentChooser', 'PREFER_ORGANIC', 'REMOVE_FRAGMENTS', 'is_organic', 'log', 'logging', 'memoized_property', 'rdMolDescriptors', 'warn']
class FragmentPattern:
    """
    A fragment defined by a SMARTS pattern.
    """
    def __init__(self, name, smarts):
        """
        Initialize a FragmentPattern with a name and a SMARTS pattern.
        
                :param name: A name for this FragmentPattern.
                :param smarts: A SMARTS pattern.
                
        """
    def __repr__(self):
        ...
    def __str__(self):
        ...
    @property
    def smarts(self):
        ...
class FragmentRemover:
    """
    A class for filtering out fragments using SMARTS patterns.
    """
    def __call__(self, mol):
        """
        Calling a FragmentRemover instance like a function is the same as calling its remove(mol) method.
        """
    def __init__(self, fragments = ..., leave_last = True):
        """
        Initialize a FragmentRemover with an optional custom list of :class:`~molvs.fragment.FragmentPattern`.
        
                Setting leave_last to True will ensure at least one fragment is left in the molecule, even if it is matched by a
                :class:`~molvs.fragment.FragmentPattern`. Fragments are removed in the order specified in the list, so place
                those you would prefer to be left towards the end of the list. If all the remaining fragments match the same
                :class:`~molvs.fragment.FragmentPattern`, they will all be left.
        
                :param fragments: A list of :class:`~molvs.fragment.FragmentPattern` to remove.
                :param bool leave_last: Whether to ensure at least one fragment is left.
                
        """
    def remove(self, mol):
        """
        Return the molecule with specified fragments removed.
        
                :param mol: The molecule to remove fragments from.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :return: The molecule with fragments removed.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
class LargestFragmentChooser:
    """
    A class for selecting the largest covalent unit in a molecule with multiple fragments.
    """
    def __call__(self, mol):
        """
        Calling a LargestFragmentChooser instance like a function is the same as calling its choose(mol) method.
        """
    def __init__(self, prefer_organic = False):
        """
        
        
                If prefer_organic is set to True, any organic fragment will be considered larger than any inorganic fragment. A
                fragment is considered organic if it contains a carbon atom.
        
                :param bool prefer_organic: Whether to prioritize organic fragments above all others.
                
        """
    def choose(self, mol):
        """
        Return the largest covalent unit.
        
                The largest fragment is determined by number of atoms (including hydrogens). Ties are broken by taking the
                fragment with the higher molecular weight, and then by taking the first alphabetically by SMILES if needed.
        
                :param mol: The molecule to choose the largest fragment from.
                :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                :return: The largest fragment.
                :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
                
        """
def is_organic(fragment):
    """
    Return true if fragment contains at least one carbon atom.
    
        :param fragment: The fragment as an RDKit Mol object.
        
    """
LEAVE_LAST: bool = True
PREFER_ORGANIC: bool = False
REMOVE_FRAGMENTS: tuple  # value = (FragmentPattern('hydrogen', '[H]'), FragmentPattern('fluorine', '[F]'), FragmentPattern('chlorine', '[Cl]'), FragmentPattern('bromine', '[Br]'), FragmentPattern('iodine', '[I]'), FragmentPattern('lithium', '[Li]'), FragmentPattern('sodium', '[Na]'), FragmentPattern('potassium', '[K]'), FragmentPattern('calcium', '[Ca]'), FragmentPattern('magnesium', '[Mg]'), FragmentPattern('aluminium', '[Al]'), FragmentPattern('barium', '[Ba]'), FragmentPattern('bismuth', '[Bi]'), FragmentPattern('silver', '[Ag]'), FragmentPattern('strontium', '[Sr]'), FragmentPattern('zinc', '[Zn]'), FragmentPattern('ammonia/ammonium', '[#7]'), FragmentPattern('water/hydroxide', '[#8]'), FragmentPattern('methyl amine', '[#6]-[#7]'), FragmentPattern('sulfide', 'S'), FragmentPattern('nitrate', '[#7](=[#8])(-[#8])-[#8]'), FragmentPattern('phosphate', '[P](=[#8])(-[#8])(-[#8])-[#8]'), FragmentPattern('hexafluorophosphate', '[P](-[#9])(-[#9])(-[#9])(-[#9])(-[#9])-[#9]'), FragmentPattern('sulfate', '[S](=[#8])(=[#8])(-[#8])-[#8]'), FragmentPattern('methyl sulfonate', '[#6]-[S](=[#8])(=[#8])(-[#8])'), FragmentPattern('trifluoromethanesulfonic acid', '[#8]-[S](=[#8])(=[#8])-[#6](-[#9])(-[#9])-[#9]'), FragmentPattern('trifluoroacetic acid', '[#9]-[#6](-[#9])(-[#9])-[#6](=[#8])-[#8]'), FragmentPattern('1,2-dichloroethane', '[Cl]-[#6]-[#6]-[Cl]'), FragmentPattern('1,2-dimethoxyethane', '[#6]-[#8]-[#6]-[#6]-[#8]-[#6]'), FragmentPattern('1,4-dioxane', '[#6]-1-[#6]-[#8]-[#6]-[#6]-[#8]-1'), FragmentPattern('1-methyl-2-pyrrolidinone', '[#6]-[#7]-1-[#6]-[#6]-[#6]-[#6]-1=[#8]'), FragmentPattern('2-butanone', '[#6]-[#6]-[#6](-[#6])=[#8]'), FragmentPattern('acetate/acetic acid', '[#8]-[#6](-[#6])=[#8]'), FragmentPattern('acetone', '[#6]-[#6](-[#6])=[#8]'), FragmentPattern('acetonitrile', '[#6]-[#6]#[N]'), FragmentPattern('benzene', '[#6]1[#6][#6][#6][#6][#6]1'), FragmentPattern('butanol', '[#8]-[#6]-[#6]-[#6]-[#6]'), FragmentPattern('t-butanol', '[#8]-[#6](-[#6])(-[#6])-[#6]'), FragmentPattern('chloroform', '[Cl]-[#6](-[Cl])-[Cl]'), FragmentPattern('cycloheptane', '[#6]-1-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-1'), FragmentPattern('cyclohexane', '[#6]-1-[#6]-[#6]-[#6]-[#6]-[#6]-1'), FragmentPattern('dichloromethane', '[Cl]-[#6]-[Cl]'), FragmentPattern('diethyl ether', '[#6]-[#6]-[#8]-[#6]-[#6]'), FragmentPattern('diisopropyl ether', '[#6]-[#6](-[#6])-[#8]-[#6](-[#6])-[#6]'), FragmentPattern('dimethyl formamide', '[#6]-[#7](-[#6])-[#6]=[#8]'), FragmentPattern('dimethyl sulfoxide', '[#6]-[S](-[#6])=[#8]'), FragmentPattern('ethanol', '[#8]-[#6]-[#6]'), FragmentPattern('ethyl acetate', '[#6]-[#6]-[#8]-[#6](-[#6])=[#8]'), FragmentPattern('formic acid', '[#8]-[#6]=[#8]'), FragmentPattern('heptane', '[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]'), FragmentPattern('hexane', '[#6]-[#6]-[#6]-[#6]-[#6]-[#6]'), FragmentPattern('isopropanol', '[#8]-[#6](-[#6])-[#6]'), FragmentPattern('methanol', '[#8]-[#6]'), FragmentPattern('N,N-dimethylacetamide', '[#6]-[#7](-[#6])-[#6](-[#6])=[#8]'), FragmentPattern('pentane', '[#6]-[#6]-[#6]-[#6]-[#6]'), FragmentPattern('propanol', '[#8]-[#6]-[#6]-[#6]'), FragmentPattern('pyridine', '[#6]-1=[#6]-[#6]=[#7]-[#6]=[#6]-1'), FragmentPattern('t-butyl methyl ether', '[#6]-[#8]-[#6](-[#6])(-[#6])-[#6]'), FragmentPattern('tetrahydrofurane', '[#6]-1-[#6]-[#6]-[#8]-[#6]-1'), FragmentPattern('toluene', '[#6]-[#6]~1~[#6]~[#6]~[#6]~[#6]~[#6]~1'), FragmentPattern('xylene', '[#6]-[#6]~1~[#6](-[#6])~[#6]~[#6]~[#6]~[#6]~1'))
log: logging.Logger  # value = <Logger rdkit.Chem.MolStandardize.fragment (WARNING)>
