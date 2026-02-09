"""

molvs.utils
~~~~~~~~~~~

This module contains miscellaneous utility functions.

:copyright: Copyright 2016 by Matt Swain.
:license: MIT, see LICENSE file for more details.
"""
from __future__ import annotations
import functools as functools
from itertools import tee
__all__ = ['functools', 'memoized_property', 'pairwise', 'tee']
def memoized_property(fget):
    """
    Decorator to create memoized properties.
    """
def pairwise(iterable):
    """
    Utility function to iterate in a pairwise fashion.
    """
