"""
Lazy loading of moved objects in six.moves.urllib_request
"""
from __future__ import annotations
from urllib.request import AbstractBasicAuthHandler
from urllib.request import AbstractDigestAuthHandler
from urllib.request import BaseHandler
from urllib.request import CacheFTPHandler
from urllib.request import FTPHandler
from urllib.request import FancyURLopener
from urllib.request import FileHandler
from urllib.request import HTTPBasicAuthHandler
from urllib.request import HTTPCookieProcessor
from urllib.request import HTTPDefaultErrorHandler
from urllib.request import HTTPDigestAuthHandler
from urllib.request import HTTPErrorProcessor
from urllib.request import HTTPHandler
from urllib.request import HTTPPasswordMgr
from urllib.request import HTTPPasswordMgrWithDefaultRealm
from urllib.request import HTTPRedirectHandler
from urllib.request import HTTPSHandler
from urllib.request import OpenerDirector
from urllib.request import ProxyBasicAuthHandler
from urllib.request import ProxyDigestAuthHandler
from urllib.request import ProxyHandler
from urllib.request import Request
from urllib.request import URLopener
from urllib.request import UnknownHandler
from urllib.request import build_opener
from urllib.request import getproxies_environment as getproxies
from urllib.request import install_opener
from urllib.request import pathname2url
from urllib.request import proxy_bypass_environment as proxy_bypass
from urllib.request import url2pathname
from urllib.request import urlcleanup
from urllib.request import urlopen
from urllib.request import urlretrieve
__all__ = ['AbstractBasicAuthHandler', 'AbstractDigestAuthHandler', 'BaseHandler', 'CacheFTPHandler', 'FTPHandler', 'FancyURLopener', 'FileHandler', 'HTTPBasicAuthHandler', 'HTTPCookieProcessor', 'HTTPDefaultErrorHandler', 'HTTPDigestAuthHandler', 'HTTPErrorProcessor', 'HTTPHandler', 'HTTPPasswordMgr', 'HTTPPasswordMgrWithDefaultRealm', 'HTTPRedirectHandler', 'HTTPSHandler', 'OpenerDirector', 'ProxyBasicAuthHandler', 'ProxyDigestAuthHandler', 'ProxyHandler', 'Request', 'URLopener', 'UnknownHandler', 'build_opener', 'getproxies', 'install_opener', 'pathname2url', 'proxy_bypass', 'url2pathname', 'urlcleanup', 'urlopen', 'urlretrieve']
