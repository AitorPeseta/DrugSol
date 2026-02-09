"""

molvs.errors
~~~~~~~~~~~~

This module contains exceptions that are raised by MolVS.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
from __future__ import annotations
__all__ = ['MolVSError', 'StandardizeError', 'StopValidateError', 'ValidateError']
class MolVSError(Exception):
    pass
class StandardizeError(MolVSError):
    pass
class StopValidateError(ValidateError):
    """
    Called by Validations to stop any further validations from being performed.
    """
class ValidateError(MolVSError):
    pass
