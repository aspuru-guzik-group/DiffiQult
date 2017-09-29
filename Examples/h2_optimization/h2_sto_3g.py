import sys
import time 
sys.path.insert(0, '../../')

from Task import autochem 
from Basis import basis_set3
 
d = -1.64601435
mol = [(1,(0.0,0.0,0.20165898)),
(1,(0.0,0.0,d))]

tasks = ["Energy","Opt","Opt"]
argnum = [[],[2,5],[2],[0],[2]]
autochem(tasks,        ## Tasks
         mol,          ## Geometry
         basis_set3,   ## Basis set (if shifted it should have the coordinates too)
         2/2,          ## Number of electrons
         shifted=False,      ## If the basis is going to be on the atoms coordinates 
         name='h2_sto_3g',   ## Prefix for all optput files
         verbose=True,       ## If there is going to be an output
         readguess=False,    ## If the optimization will use an initial guess of C
         argnum=argnum,## Order of optimizations ([], for energy)
         maxiter=5)         ## Maximum number of steps in the optimization
