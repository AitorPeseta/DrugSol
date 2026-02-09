"""
Utilities for writing code that runs on Python 2 and 3
"""
from __future__ import annotations
from _io import BytesIO
from _io import StringIO
from _operator import getitem as indexbytes
from builtins import bytes as binary_type
from builtins import callable
from builtins import chr as unichr
from builtins import exec as exec_
from builtins import iter as iterbytes
from builtins import method as create_bound_method
from builtins import next
from builtins import next as advance_iterator
from builtins import object as Iterator
from builtins import print as print_
from builtins import str as text_type
import io as io
import operator as operator
import sys as sys
import types as types
import typing
from .moves import *
__all__ = ['BytesIO', 'Iterator', 'MAXSIZE', 'Module_six_moves_urllib', 'Module_six_moves_urllib_error', 'Module_six_moves_urllib_parse', 'Module_six_moves_urllib_request', 'Module_six_moves_urllib_response', 'Module_six_moves_urllib_robotparser', 'MovedAttribute', 'MovedModule', 'PY2', 'PY3', 'StringIO', 'add_metaclass', 'add_move', 'advance_iterator', 'b', 'binary_type', 'byte2int', 'callable', 'class_types', 'cmp', 'create_bound_method', 'exec_', 'get_function_closure', 'get_function_code', 'get_function_defaults', 'get_function_globals', 'get_method_function', 'get_method_self', 'get_unbound_function', 'indexbytes', 'int2byte', 'integer_types', 'io', 'iterbytes', 'iteritems', 'iterkeys', 'iterlists', 'itervalues', 'moves', 'next', 'operator', 'print_', 'remove_move', 'reraise', 'string_types', 'sys', 'text_type', 'types', 'u', 'unichr', 'with_metaclass']
class Module_six_moves_urllib(module):
    """
    Create a six.moves.urllib namespace that resembles the Python 3 namespace
    """
    error = moves.urllib.error
    parse = moves.urllib_parse
    request = moves.urllib.request
    response = moves.urllib.response
    robotparser = moves.urllib.robotparser
class Module_six_moves_urllib_error(_LazyModule):
    """
    Lazy loading of moved objects in six.moves.urllib_error
    """
    _moved_attributes: typing.ClassVar[list]  # value = [<rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>]
    @staticmethod
    def ContentTooShortError(*args, **kwargs):
        ...
    @staticmethod
    def HTTPError(*args, **kwargs):
        ...
    @staticmethod
    def URLError(*args, **kwargs):
        ...
class Module_six_moves_urllib_parse(_LazyModule):
    """
    Lazy loading of moved objects in six.moves.urllib_parse
    """
    _moved_attributes: typing.ClassVar[list]  # value = [<rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>]
    @staticmethod
    def ParseResult(*args, **kwargs):
        ...
    @staticmethod
    def SplitResult(*args, **kwargs):
        ...
    @staticmethod
    def parse_qs(*args, **kwargs):
        ...
    @staticmethod
    def parse_qsl(*args, **kwargs):
        ...
    @staticmethod
    def quote(*args, **kwargs):
        ...
    @staticmethod
    def quote_plus(*args, **kwargs):
        ...
    @staticmethod
    def splitquery(*args, **kwargs):
        ...
    @staticmethod
    def unquote(*args, **kwargs):
        ...
    @staticmethod
    def unquote_plus(*args, **kwargs):
        ...
    @staticmethod
    def urldefrag(*args, **kwargs):
        ...
    @staticmethod
    def urlencode(*args, **kwargs):
        ...
    @staticmethod
    def urljoin(*args, **kwargs):
        ...
    @staticmethod
    def urlparse(*args, **kwargs):
        ...
    @staticmethod
    def urlsplit(*args, **kwargs):
        ...
    @staticmethod
    def urlunparse(*args, **kwargs):
        ...
    @staticmethod
    def urlunsplit(*args, **kwargs):
        ...
class Module_six_moves_urllib_request(_LazyModule):
    """
    Lazy loading of moved objects in six.moves.urllib_request
    """
    _moved_attributes: typing.ClassVar[list]  # value = [<rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>]
    @staticmethod
    def AbstractBasicAuthHandler(*args, **kwargs):
        ...
    @staticmethod
    def AbstractDigestAuthHandler(*args, **kwargs):
        ...
    @staticmethod
    def BaseHandler(*args, **kwargs):
        ...
    @staticmethod
    def CacheFTPHandler(*args, **kwargs):
        ...
    @staticmethod
    def FTPHandler(*args, **kwargs):
        ...
    @staticmethod
    def FancyURLopener(*args, **kwargs):
        ...
    @staticmethod
    def FileHandler(*args, **kwargs):
        ...
    @staticmethod
    def HTTPBasicAuthHandler(*args, **kwargs):
        ...
    @staticmethod
    def HTTPCookieProcessor(*args, **kwargs):
        ...
    @staticmethod
    def HTTPDefaultErrorHandler(*args, **kwargs):
        ...
    @staticmethod
    def HTTPDigestAuthHandler(*args, **kwargs):
        ...
    @staticmethod
    def HTTPErrorProcessor(*args, **kwargs):
        ...
    @staticmethod
    def HTTPHandler(*args, **kwargs):
        ...
    @staticmethod
    def HTTPPasswordMgr(*args, **kwargs):
        ...
    @staticmethod
    def HTTPPasswordMgrWithDefaultRealm(*args, **kwargs):
        ...
    @staticmethod
    def HTTPRedirectHandler(*args, **kwargs):
        ...
    @staticmethod
    def HTTPSHandler(*args, **kwargs):
        ...
    @staticmethod
    def OpenerDirector(*args, **kwargs):
        ...
    @staticmethod
    def ProxyBasicAuthHandler(*args, **kwargs):
        ...
    @staticmethod
    def ProxyDigestAuthHandler(*args, **kwargs):
        ...
    @staticmethod
    def ProxyHandler(*args, **kwargs):
        ...
    @staticmethod
    def Request(*args, **kwargs):
        ...
    @staticmethod
    def URLopener(*args, **kwargs):
        ...
    @staticmethod
    def UnknownHandler(*args, **kwargs):
        ...
    @staticmethod
    def build_opener(*args, **kwargs):
        ...
    @staticmethod
    def getproxies(*args, **kwargs):
        ...
    @staticmethod
    def install_opener(*args, **kwargs):
        ...
    @staticmethod
    def pathname2url(*args, **kwargs):
        ...
    @staticmethod
    def proxy_bypass(*args, **kwargs):
        ...
    @staticmethod
    def url2pathname(*args, **kwargs):
        ...
    @staticmethod
    def urlcleanup(*args, **kwargs):
        ...
    @staticmethod
    def urlopen(*args, **kwargs):
        ...
    @staticmethod
    def urlretrieve(*args, **kwargs):
        ...
class Module_six_moves_urllib_response(_LazyModule):
    """
    Lazy loading of moved objects in six.moves.urllib_response
    """
    _moved_attributes: typing.ClassVar[list]  # value = [<rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>]
    @staticmethod
    def addbase(*args, **kwargs):
        ...
    @staticmethod
    def addclosehook(*args, **kwargs):
        ...
    @staticmethod
    def addinfo(*args, **kwargs):
        ...
    @staticmethod
    def addinfourl(*args, **kwargs):
        ...
class Module_six_moves_urllib_robotparser(_LazyModule):
    """
    Lazy loading of moved objects in six.moves.urllib_robotparser
    """
    _moved_attributes: typing.ClassVar[list]  # value = [<rdkit.six.MovedAttribute object>]
    @staticmethod
    def RobotFileParser(*args, **kwargs):
        ...
class MovedAttribute(_LazyDescr):
    def __init__(self, name, old_mod, new_mod, old_attr = None, new_attr = None):
        ...
    def _resolve(self):
        ...
class MovedModule(_LazyDescr):
    def __getattr__(self, attr):
        ...
    def __init__(self, name, old, new = None):
        ...
    def _resolve(self):
        ...
class _LazyDescr:
    def __get__(self, obj, tp):
        ...
    def __init__(self, name):
        ...
class _LazyModule(module):
    _moved_attributes: typing.ClassVar[list] = list()
    def __init__(self, name):
        ...
class _MovedItems(_LazyModule):
    """
    Lazy loading of moved objects
    """
    _moved_attributes: typing.ClassVar[list]  # value = [<rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>]
    @staticmethod
    def BaseHTTPServer(*args, **kwargs):
        ...
    @staticmethod
    def CGIHTTPServer(*args, **kwargs):
        ...
    @staticmethod
    def SimpleHTTPServer(*args, **kwargs):
        ...
    @staticmethod
    def StringIO(*args, **kwargs):
        ...
    @staticmethod
    def UserDict(*args, **kwargs):
        ...
    @staticmethod
    def UserList(*args, **kwargs):
        ...
    @staticmethod
    def UserString(*args, **kwargs):
        ...
    @staticmethod
    def _thread(*args, **kwargs):
        ...
    @staticmethod
    def cPickle(*args, **kwargs):
        ...
    @staticmethod
    def cStringIO(*args, **kwargs):
        ...
    @staticmethod
    def configparser(*args, **kwargs):
        ...
    @staticmethod
    def copyreg(*args, **kwargs):
        ...
    @staticmethod
    def dbm_gnu(*args, **kwargs):
        ...
    @staticmethod
    def email_mime_base(*args, **kwargs):
        ...
    @staticmethod
    def email_mime_multipart(*args, **kwargs):
        ...
    @staticmethod
    def email_mime_text(*args, **kwargs):
        ...
    @staticmethod
    def filter(*args, **kwargs):
        ...
    @staticmethod
    def filterfalse(*args, **kwargs):
        ...
    @staticmethod
    def html_entities(*args, **kwargs):
        ...
    @staticmethod
    def html_parser(*args, **kwargs):
        ...
    @staticmethod
    def http_client(*args, **kwargs):
        ...
    @staticmethod
    def http_cookiejar(*args, **kwargs):
        ...
    @staticmethod
    def http_cookies(*args, **kwargs):
        ...
    @staticmethod
    def input(*args, **kwargs):
        ...
    @staticmethod
    def map(*args, **kwargs):
        ...
    @staticmethod
    def queue(*args, **kwargs):
        ...
    @staticmethod
    def range(*args, **kwargs):
        ...
    @staticmethod
    def reduce(*args, **kwargs):
        ...
    @staticmethod
    def reload_module(*args, **kwargs):
        ...
    @staticmethod
    def reprlib(*args, **kwargs):
        ...
    @staticmethod
    def socketserver(*args, **kwargs):
        ...
    @staticmethod
    def tkinter(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_colorchooser(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_commondialog(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_constants(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_dialog(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_dnd(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_filedialog(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_font(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_messagebox(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_scrolledtext(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_simpledialog(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_tix(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_tkfiledialog(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_tksimpledialog(*args, **kwargs):
        ...
    @staticmethod
    def tkinter_ttk(*args, **kwargs):
        ...
    @staticmethod
    def urllib(*args, **kwargs):
        ...
    @staticmethod
    def urllib_error(*args, **kwargs):
        ...
    @staticmethod
    def urllib_parse(*args, **kwargs):
        ...
    @staticmethod
    def urllib_robotparser(*args, **kwargs):
        ...
    @staticmethod
    def winreg(*args, **kwargs):
        ...
    @staticmethod
    def xmlrpc_client(*args, **kwargs):
        ...
    @staticmethod
    def xmlrpc_server(*args, **kwargs):
        ...
    @staticmethod
    def xrange(*args, **kwargs):
        ...
    @staticmethod
    def zip(*args, **kwargs):
        ...
    @staticmethod
    def zip_longest(*args, **kwargs):
        ...
def _add_doc(func, doc):
    """
    Add documentation to a function.
    """
def _import_module(name):
    """
    Import module, returning the module after the last dot.
    """
def add_metaclass(metaclass):
    """
    Class decorator for creating a class with a metaclass.
    """
def add_move(move):
    """
    Add an item to six.moves.
    """
def b(s):
    """
    Byte literal
    """
def cmp(t1, t2):
    ...
def get_unbound_function(unbound):
    """
    Get the function out of a possibly unbound function
    """
def iteritems(d, **kw):
    """
    Return an iterator over the (key, value) pairs of a dictionary.
    """
def iterkeys(d, **kw):
    """
    Return an iterator over the keys of a dictionary.
    """
def iterlists(d, **kw):
    """
    Return an iterator over the (key, [values]) pairs of a dictionary.
    """
def itervalues(d, **kw):
    """
    Return an iterator over the values of a dictionary.
    """
def remove_move(name):
    """
    Remove item from six.moves.
    """
def reraise(tp, value, tb = None):
    """
    Reraise an exception.
    """
def u(s):
    """
    Text literal
    """
def with_metaclass(meta, *bases):
    """
    Create a base class with a metaclass.
    """
MAXSIZE: int = 9223372036854775807
PY2: bool = False
PY3: bool = True
__author__: str = 'Benjamin Peterson <benjamin@python.org>'
__version__: str = '1.6.1'
_func_closure: str = '__closure__'
_func_code: str = '__code__'
_func_defaults: str = '__defaults__'
_func_globals: str = '__globals__'
_meth_func: str = '__func__'
_meth_self: str = '__self__'
_moved_attributes: list  # value = [<rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>, <rdkit.six.MovedModule object>]
_urllib_error_moved_attributes: list  # value = [<rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>]
_urllib_parse_moved_attributes: list  # value = [<rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>]
_urllib_request_moved_attributes: list  # value = [<rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>]
_urllib_response_moved_attributes: list  # value = [<rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>, <rdkit.six.MovedAttribute object>]
_urllib_robotparser_moved_attributes: list  # value = [<rdkit.six.MovedAttribute object>]
byte2int: operator.itemgetter  # value = operator.itemgetter(0)
class_types: tuple = (type)
get_function_closure: operator.attrgetter  # value = operator.attrgetter('__closure__')
get_function_code: operator.attrgetter  # value = operator.attrgetter('__code__')
get_function_defaults: operator.attrgetter  # value = operator.attrgetter('__defaults__')
get_function_globals: operator.attrgetter  # value = operator.attrgetter('__globals__')
get_method_function: operator.attrgetter  # value = operator.attrgetter('__func__')
get_method_self: operator.attrgetter  # value = operator.attrgetter('__self__')
int2byte: operator.methodcaller  # value = operator.methodcaller('to_bytes', 1, 'big')
integer_types: tuple = (int)
string_types: tuple = (str)
