"""
 contains a class to store parameters for and results from
Composite building

"""
from __future__ import annotations
from _warnings import warn
from rdkit.Dbase.DbConnection import DbConnect
from rdkit.Dbase import DbModule
from rdkit import RDConfig
import typing
__all__ = ['CompositeRun', 'DbConnect', 'DbModule', 'RDConfig', 'SetDefaults', 'warn']
class CompositeRun:
    """
     class to store parameters for and results from Composite building
    
       This class has a default set of fields which are added to the database.
    
       By default these fields are stored in a tuple, so they are immutable.  This
         is probably what you want.
    
    
      
    """
    fields: typing.ClassVar[tuple] = (('rundate', 'varchar(32)'), ('dbName', 'varchar(200)'), ('dbWhat', 'varchar(200)'), ('dbWhere', 'varchar(200)'), ('dbJoin', 'varchar(200)'), ('tableName', 'varchar(80)'), ('note', 'varchar(120)'), ('shuffled', 'smallint'), ('randomized', 'smallint'), ('overall_error', 'float'), ('holdout_error', 'float'), ('overall_fraction_dropped', 'float'), ('holdout_fraction_dropped', 'float'), ('overall_correct_conf', 'float'), ('overall_incorrect_conf', 'float'), ('holdout_correct_conf', 'float'), ('holdout_incorrect_conf', 'float'), ('overall_result_matrix', 'varchar(256)'), ('holdout_result_matrix', 'varchar(256)'), ('threshold', 'float'), ('splitFrac', 'float'), ('filterFrac', 'float'), ('filterVal', 'float'), ('modelFilterVal', 'float'), ('modelFilterFrac', 'float'), ('nModels', 'int'), ('limitDepth', 'int'), ('bayesModels', 'int'), ('qBoundCount', 'varchar(3000)'), ('activityBoundsVals', 'varchar(200)'), ('cmd', 'varchar(500)'), ('model', 'blob'))
    def GetDataSet(self, **kwargs):
        """
         Returns a MLDataSet pulled from a database using our stored
            values.
        
            
        """
    def GetDataSetInfo(self, **kwargs):
        """
         Returns a MLDataSet pulled from a database using our stored
            values.
        
            
        """
    def Store(self, db = 'models.gdb', table = 'results', user = 'sysdba', password = 'masterkey'):
        """
         adds the result to a database
        
              **Arguments**
        
                - db: name of the database to use
        
                - table: name of the table to use
        
                - user&password: connection information
        
            
        """
    def _CreateTable(self, cn, tblName):
        """
         *Internal Use only*
        
            
        """
def SetDefaults(runDetails):
    """
      initializes a details object with default values
    
          **Arguments**
    
            - details:  (optional) a _CompositeRun.CompositeRun_ object.
              If this is not provided, the global _runDetails will be used.
    
          **Returns**
    
            the initialized _CompositeRun_ object.
    
    
      
    """
