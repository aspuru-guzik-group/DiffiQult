import sys
import time 
sys.path.insert(0, '../../')

from Task import autochem 
from Basis import sto3g_contracted
 
d = -1.64601435
mol = [(9,(0.0, 0.0,0.20165898)),
     (1,(0.0, 0.0,-1.24601435))]


tasks = ["Opt","Opt"]
argnum = [[0,1],[0],[2]]
autochem(tasks,        ## Tasks
         mol,          ## Geometry
         sto3g_contracted,   ## Basis set (if shifted it should have the coordinates too)
         10/2,          ## Number of electrons
         shifted=False,      ## If the basis is going to be on the atoms coordinates 
         name='hf_sto_3gc',   ## Prefix for all optput files
         verbose=True,       ## If there is going to be an output
         readguess=False,    ## If the optimization will use an initial guess of C
         argnum=argnum,## Order of optimizations ([], for energy)
         maxiter=5)         ## Maximum number of steps in the optimization
