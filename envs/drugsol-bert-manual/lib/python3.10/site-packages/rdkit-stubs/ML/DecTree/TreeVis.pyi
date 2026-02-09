"""
 functionality for drawing trees on sping canvases

"""
from __future__ import annotations
import math as math
import rdkit.sping.colors
import rdkit.sping.pid
from rdkit.sping import pid as piddle
import typing
__all__ = ['CalcTreeNodeSizes', 'CalcTreeWidth', 'DrawTree', 'DrawTreeNode', 'ResetTree', 'SetNodeScales', 'VisOpts', 'math', 'piddle', 'visOpts']
class VisOpts:
    circColor: typing.ClassVar[rdkit.sping.colors.Color]  # value = Color(0.60,0.60,0.90)
    circRad: typing.ClassVar[int] = 10
    highlightColor: typing.ClassVar[rdkit.sping.colors.Color]  # value = Color(1.00,1.00,0.40)
    highlightWidth: typing.ClassVar[int] = 2
    horizOffset: typing.ClassVar[int] = 10
    labelFont: typing.ClassVar[rdkit.sping.pid.Font]  # value = Font(10,0,0,0,'helvetica')
    lineColor: typing.ClassVar[rdkit.sping.colors.Color]  # value = Color(0.00,0.00,0.00)
    lineWidth: typing.ClassVar[int] = 2
    maxCircRad: typing.ClassVar[int] = 16
    minCircRad: typing.ClassVar[int] = 4
    outlineColor: typing.ClassVar[rdkit.sping.colors.Color]  # value = Color(-1.00,-1.00,-1.00)
    terminalEmptyColor: typing.ClassVar[rdkit.sping.colors.Color]  # value = Color(0.80,0.80,0.20)
    terminalOffColor: typing.ClassVar[rdkit.sping.colors.Color]  # value = Color(0.20,0.20,0.20)
    terminalOnColor: typing.ClassVar[rdkit.sping.colors.Color]  # value = Color(0.80,0.80,0.80)
    vertOffset: typing.ClassVar[int] = 50
def CalcTreeNodeSizes(node):
    """
    Recursively calculate the total number of nodes under us.
    
        results are set in node.totNChildren for this node and
        everything underneath it.
      
    """
def CalcTreeWidth(tree):
    ...
def DrawTree(tree, canvas, nRes = 2, scaleLeaves = False, allowShrink = True, showPurity = False):
    ...
def DrawTreeNode(node, loc, canvas, nRes = 2, scaleLeaves = False, showPurity = False):
    """
    Recursively displays the given tree node and all its children on the canvas
      
    """
def ResetTree(tree):
    ...
def SetNodeScales(node):
    ...
def _ApplyNodeScales(node, min, max):
    ...
def _ExampleCounter(node, min, max):
    ...
def _simpleTest(canv):
    ...
visOpts: VisOpts  # value = <rdkit.ML.DecTree.TreeVis.VisOpts object>
