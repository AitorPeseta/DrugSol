"""
Lazy loading of moved objects in six.moves.urllib_response
"""
from __future__ import annotations
from urllib.response import addbase
from urllib.response import addclosehook
from urllib.response import addinfo
from urllib.response import addinfourl
__all__ = ['addbase', 'addclosehook', 'addinfo', 'addinfourl']
