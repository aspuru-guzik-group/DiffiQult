import sys
import time 
sys.path.insert(0, '../../')

from Task import autochem 
from Basis import basis_set2
 
d = -1.64601435
mol = [
( 8,(0.0, 0.0, 0.091685801102911746)),
( 1,(1.4229678834888837, 0.0, -0.98120954931681137)),
( 1,(-1.4229678834888837, 0.0, -0.98120954931681137))]


tasks = ["Opt"]
tasks = ["Energy","Opt","Opt"]
argnum = [[],[0],[2],[0],[2]]
autochem(tasks,        ## Tasks
         mol,          ## Geometry
         basis_set2,   ## Basis set (if shifted it should have the coordinates too)
         10/2,          ## Number of electrons
         shifted=False,      ## If the basis is going to be on the atoms coordinates 
         name='h2o_sto_2g',   ## Prefix for all optput files
         verbose=True,       ## If there is going to be an output
         readguess=False,    ## If the optimization will use an initial guess of C
         argnum=[[],[0],[2]],## Order of optimizations ([], for energy)
         maxiter=5,
         output=True)         ## Maximum number of steps in the optimization
