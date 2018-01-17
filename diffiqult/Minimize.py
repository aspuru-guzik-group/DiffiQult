"""
This module was originally a scipy module
Functions
---------
"""
import Optimize
from Optimize import minimize_bfgs

## TODO add tool value
def minimize(fun, x0, args=(),  method='BFGS', jac=None,tol=None,gtol=None,
             callback=None, argnum=None,name=None,**options):

    if method == 'BFGS':
        return minimize_bfgs(fun, x0, args, argnum, jac, callback, name, gtol,**options)
    else:
        raise ValueError('Unknown solver %s' % method)


