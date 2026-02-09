"""

molvs.validations
~~~~~~~~~~~~~~~~~

This module contains all the built-in :class:`Validations <molvs.validations.Validation>`.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
from __future__ import annotations
from _warnings import warn
import logging as logging
from rdkit import Chem
from rdkit.Chem.MolStandardize.errors import StopValidateError
import typing
__all__ = ['Chem', 'DichloroethaneValidation', 'FragmentValidation', 'IsNoneValidation', 'IsotopeValidation', 'NeutralValidation', 'NoAtomValidation', 'REMOVE_FRAGMENTS', 'SmartsValidation', 'StopValidateError', 'VALIDATIONS', 'Validation', 'logging', 'warn']
class DichloroethaneValidation(SmartsValidation):
    """
    Logs if 1,2-dichloroethane is present.
    
        This is provided as an example of how to subclass :class:`~molvs.validations.SmartsValidation` to check for the
        presence of a substructure.
        
    """
    entire_fragment: typing.ClassVar[bool] = True
    level: typing.ClassVar[int] = 20
    message: typing.ClassVar[str] = '1,2-Dichloroethane is present'
    smarts: typing.ClassVar[str] = '[Cl]-[#6]-[#6]-[Cl]'
class FragmentValidation(Validation):
    """
    Logs if certain fragments are present.
    
        Subclass and override the ``fragments`` class attribute to customize the list of
        :class:`FragmentPatterns <molvs.fragment.FragmentPattern>`.
        
    """
    fragments: typing.ClassVar[tuple]  # value = (FragmentPattern('hydrogen', '[H]'), FragmentPattern('fluorine', '[F]'), FragmentPattern('chlorine', '[Cl]'), FragmentPattern('bromine', '[Br]'), FragmentPattern('iodine', '[I]'), FragmentPattern('lithium', '[Li]'), FragmentPattern('sodium', '[Na]'), FragmentPattern('potassium', '[K]'), FragmentPattern('calcium', '[Ca]'), FragmentPattern('magnesium', '[Mg]'), FragmentPattern('aluminium', '[Al]'), FragmentPattern('barium', '[Ba]'), FragmentPattern('bismuth', '[Bi]'), FragmentPattern('silver', '[Ag]'), FragmentPattern('strontium', '[Sr]'), FragmentPattern('zinc', '[Zn]'), FragmentPattern('ammonia/ammonium', '[#7]'), FragmentPattern('water/hydroxide', '[#8]'), FragmentPattern('methyl amine', '[#6]-[#7]'), FragmentPattern('sulfide', 'S'), FragmentPattern('nitrate', '[#7](=[#8])(-[#8])-[#8]'), FragmentPattern('phosphate', '[P](=[#8])(-[#8])(-[#8])-[#8]'), FragmentPattern('hexafluorophosphate', '[P](-[#9])(-[#9])(-[#9])(-[#9])(-[#9])-[#9]'), FragmentPattern('sulfate', '[S](=[#8])(=[#8])(-[#8])-[#8]'), FragmentPattern('methyl sulfonate', '[#6]-[S](=[#8])(=[#8])(-[#8])'), FragmentPattern('trifluoromethanesulfonic acid', '[#8]-[S](=[#8])(=[#8])-[#6](-[#9])(-[#9])-[#9]'), FragmentPattern('trifluoroacetic acid', '[#9]-[#6](-[#9])(-[#9])-[#6](=[#8])-[#8]'), FragmentPattern('1,2-dichloroethane', '[Cl]-[#6]-[#6]-[Cl]'), FragmentPattern('1,2-dimethoxyethane', '[#6]-[#8]-[#6]-[#6]-[#8]-[#6]'), FragmentPattern('1,4-dioxane', '[#6]-1-[#6]-[#8]-[#6]-[#6]-[#8]-1'), FragmentPattern('1-methyl-2-pyrrolidinone', '[#6]-[#7]-1-[#6]-[#6]-[#6]-[#6]-1=[#8]'), FragmentPattern('2-butanone', '[#6]-[#6]-[#6](-[#6])=[#8]'), FragmentPattern('acetate/acetic acid', '[#8]-[#6](-[#6])=[#8]'), FragmentPattern('acetone', '[#6]-[#6](-[#6])=[#8]'), FragmentPattern('acetonitrile', '[#6]-[#6]#[N]'), FragmentPattern('benzene', '[#6]1[#6][#6][#6][#6][#6]1'), FragmentPattern('butanol', '[#8]-[#6]-[#6]-[#6]-[#6]'), FragmentPattern('t-butanol', '[#8]-[#6](-[#6])(-[#6])-[#6]'), FragmentPattern('chloroform', '[Cl]-[#6](-[Cl])-[Cl]'), FragmentPattern('cycloheptane', '[#6]-1-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-1'), FragmentPattern('cyclohexane', '[#6]-1-[#6]-[#6]-[#6]-[#6]-[#6]-1'), FragmentPattern('dichloromethane', '[Cl]-[#6]-[Cl]'), FragmentPattern('diethyl ether', '[#6]-[#6]-[#8]-[#6]-[#6]'), FragmentPattern('diisopropyl ether', '[#6]-[#6](-[#6])-[#8]-[#6](-[#6])-[#6]'), FragmentPattern('dimethyl formamide', '[#6]-[#7](-[#6])-[#6]=[#8]'), FragmentPattern('dimethyl sulfoxide', '[#6]-[S](-[#6])=[#8]'), FragmentPattern('ethanol', '[#8]-[#6]-[#6]'), FragmentPattern('ethyl acetate', '[#6]-[#6]-[#8]-[#6](-[#6])=[#8]'), FragmentPattern('formic acid', '[#8]-[#6]=[#8]'), FragmentPattern('heptane', '[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]'), FragmentPattern('hexane', '[#6]-[#6]-[#6]-[#6]-[#6]-[#6]'), FragmentPattern('isopropanol', '[#8]-[#6](-[#6])-[#6]'), FragmentPattern('methanol', '[#8]-[#6]'), FragmentPattern('N,N-dimethylacetamide', '[#6]-[#7](-[#6])-[#6](-[#6])=[#8]'), FragmentPattern('pentane', '[#6]-[#6]-[#6]-[#6]-[#6]'), FragmentPattern('propanol', '[#8]-[#6]-[#6]-[#6]'), FragmentPattern('pyridine', '[#6]-1=[#6]-[#6]=[#7]-[#6]=[#6]-1'), FragmentPattern('t-butyl methyl ether', '[#6]-[#8]-[#6](-[#6])(-[#6])-[#6]'), FragmentPattern('tetrahydrofurane', '[#6]-1-[#6]-[#6]-[#8]-[#6]-1'), FragmentPattern('toluene', '[#6]-[#6]~1~[#6]~[#6]~[#6]~[#6]~[#6]~1'), FragmentPattern('xylene', '[#6]-[#6]~1~[#6](-[#6])~[#6]~[#6]~[#6]~[#6]~1'))
    def run(self, mol):
        ...
class IsNoneValidation(Validation):
    """
    Logs an error if ``None`` is passed to the Validator.
    
        This can happen if RDKit failed to parse an input format. If the molecule is ``None``, no subsequent validations
        will run.
        
    """
    def run(self, mol):
        ...
class IsotopeValidation(Validation):
    """
    Logs if molecule contains isotopes.
    """
    def run(self, mol):
        ...
class NeutralValidation(Validation):
    """
    Logs if not an overall neutral system.
    """
    def run(self, mol):
        ...
class NoAtomValidation(Validation):
    """
    Logs an error if the molecule has zero atoms.
    
        If the molecule has no atoms, no subsequent validations will run.
        
    """
    def run(self, mol):
        ...
class SmartsValidation(Validation):
    """
    Abstract superclass for :class:`Validations <molvs.validations.Validation>` that log a message if a SMARTS
        pattern matches the molecule.
    
        Subclasses can override the following attributes:
        
    """
    entire_fragment: typing.ClassVar[bool] = False
    level: typing.ClassVar[int] = 20
    message: typing.ClassVar[str] = 'Molecule matched %(smarts)s'
    def __init__(self, log):
        ...
    def _check_matches(self, mol):
        ...
    def _check_matches_fragment(self, mol):
        ...
    def run(self, mol):
        ...
    @property
    def smarts(self):
        """
        The SMARTS pattern as a string. Subclasses must implement this.
        """
class Validation:
    """
    The base class that all :class:`~molvs.validations.Validation` subclasses must inherit from.
    """
    def __call__(self, mol):
        ...
    def __init__(self, log):
        ...
    def run(self, mol):
        """
        """
REMOVE_FRAGMENTS: tuple  # value = (FragmentPattern('hydrogen', '[H]'), FragmentPattern('fluorine', '[F]'), FragmentPattern('chlorine', '[Cl]'), FragmentPattern('bromine', '[Br]'), FragmentPattern('iodine', '[I]'), FragmentPattern('lithium', '[Li]'), FragmentPattern('sodium', '[Na]'), FragmentPattern('potassium', '[K]'), FragmentPattern('calcium', '[Ca]'), FragmentPattern('magnesium', '[Mg]'), FragmentPattern('aluminium', '[Al]'), FragmentPattern('barium', '[Ba]'), FragmentPattern('bismuth', '[Bi]'), FragmentPattern('silver', '[Ag]'), FragmentPattern('strontium', '[Sr]'), FragmentPattern('zinc', '[Zn]'), FragmentPattern('ammonia/ammonium', '[#7]'), FragmentPattern('water/hydroxide', '[#8]'), FragmentPattern('methyl amine', '[#6]-[#7]'), FragmentPattern('sulfide', 'S'), FragmentPattern('nitrate', '[#7](=[#8])(-[#8])-[#8]'), FragmentPattern('phosphate', '[P](=[#8])(-[#8])(-[#8])-[#8]'), FragmentPattern('hexafluorophosphate', '[P](-[#9])(-[#9])(-[#9])(-[#9])(-[#9])-[#9]'), FragmentPattern('sulfate', '[S](=[#8])(=[#8])(-[#8])-[#8]'), FragmentPattern('methyl sulfonate', '[#6]-[S](=[#8])(=[#8])(-[#8])'), FragmentPattern('trifluoromethanesulfonic acid', '[#8]-[S](=[#8])(=[#8])-[#6](-[#9])(-[#9])-[#9]'), FragmentPattern('trifluoroacetic acid', '[#9]-[#6](-[#9])(-[#9])-[#6](=[#8])-[#8]'), FragmentPattern('1,2-dichloroethane', '[Cl]-[#6]-[#6]-[Cl]'), FragmentPattern('1,2-dimethoxyethane', '[#6]-[#8]-[#6]-[#6]-[#8]-[#6]'), FragmentPattern('1,4-dioxane', '[#6]-1-[#6]-[#8]-[#6]-[#6]-[#8]-1'), FragmentPattern('1-methyl-2-pyrrolidinone', '[#6]-[#7]-1-[#6]-[#6]-[#6]-[#6]-1=[#8]'), FragmentPattern('2-butanone', '[#6]-[#6]-[#6](-[#6])=[#8]'), FragmentPattern('acetate/acetic acid', '[#8]-[#6](-[#6])=[#8]'), FragmentPattern('acetone', '[#6]-[#6](-[#6])=[#8]'), FragmentPattern('acetonitrile', '[#6]-[#6]#[N]'), FragmentPattern('benzene', '[#6]1[#6][#6][#6][#6][#6]1'), FragmentPattern('butanol', '[#8]-[#6]-[#6]-[#6]-[#6]'), FragmentPattern('t-butanol', '[#8]-[#6](-[#6])(-[#6])-[#6]'), FragmentPattern('chloroform', '[Cl]-[#6](-[Cl])-[Cl]'), FragmentPattern('cycloheptane', '[#6]-1-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-1'), FragmentPattern('cyclohexane', '[#6]-1-[#6]-[#6]-[#6]-[#6]-[#6]-1'), FragmentPattern('dichloromethane', '[Cl]-[#6]-[Cl]'), FragmentPattern('diethyl ether', '[#6]-[#6]-[#8]-[#6]-[#6]'), FragmentPattern('diisopropyl ether', '[#6]-[#6](-[#6])-[#8]-[#6](-[#6])-[#6]'), FragmentPattern('dimethyl formamide', '[#6]-[#7](-[#6])-[#6]=[#8]'), FragmentPattern('dimethyl sulfoxide', '[#6]-[S](-[#6])=[#8]'), FragmentPattern('ethanol', '[#8]-[#6]-[#6]'), FragmentPattern('ethyl acetate', '[#6]-[#6]-[#8]-[#6](-[#6])=[#8]'), FragmentPattern('formic acid', '[#8]-[#6]=[#8]'), FragmentPattern('heptane', '[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]'), FragmentPattern('hexane', '[#6]-[#6]-[#6]-[#6]-[#6]-[#6]'), FragmentPattern('isopropanol', '[#8]-[#6](-[#6])-[#6]'), FragmentPattern('methanol', '[#8]-[#6]'), FragmentPattern('N,N-dimethylacetamide', '[#6]-[#7](-[#6])-[#6](-[#6])=[#8]'), FragmentPattern('pentane', '[#6]-[#6]-[#6]-[#6]-[#6]'), FragmentPattern('propanol', '[#8]-[#6]-[#6]-[#6]'), FragmentPattern('pyridine', '[#6]-1=[#6]-[#6]=[#7]-[#6]=[#6]-1'), FragmentPattern('t-butyl methyl ether', '[#6]-[#8]-[#6](-[#6])(-[#6])-[#6]'), FragmentPattern('tetrahydrofurane', '[#6]-1-[#6]-[#6]-[#8]-[#6]-1'), FragmentPattern('toluene', '[#6]-[#6]~1~[#6]~[#6]~[#6]~[#6]~[#6]~1'), FragmentPattern('xylene', '[#6]-[#6]~1~[#6](-[#6])~[#6]~[#6]~[#6]~[#6]~1'))
VALIDATIONS: tuple = (IsNoneValidation, NoAtomValidation, FragmentValidation, NeutralValidation, IsotopeValidation)
