"""
Lazy loading of moved objects in six.moves.urllib_error
"""
from __future__ import annotations
from urllib.error import ContentTooShortError
from urllib.error import HTTPError
from urllib.error import URLError
__all__ = ['ContentTooShortError', 'HTTPError', 'URLError']
