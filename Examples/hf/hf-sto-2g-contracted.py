import sys
sys.path.insert(0, '../../')

from Task import autochem 
from Basis import basis_set2_contracted as basis
import time

hf = [      (9,(0.0, 0.0,0.12532478)),
      (1,(0.0, 0.0,-1.56968015))]

t0 = time.clock()
tasks = ['Energy','Opt','Opt','Opt','Opt']
tasks = ['Energy']#,'Opt','Opt','Opt','Opt']
argnum = [[],[0],[2],[0],[2]]
name = 'hf-sto-2g'
autochem(tasks,        ## Tasks
         hf,          ## Geometry
         basis,   ## Basis set (if shifted it should have the coordinates too)
         10/2,          ## Number of electrons
         shifted=False,      ## If the basis is going to be on the atoms coordinates
         name=name,   ## Prefix for all optput files
         verbose=True,       ## If there is going to be an output
         readguess=True,    ## If the optimization will use an initial guess of C
         argnum=argnum,## Order of optimizations ([], for energy)
         maxiter=20)         ## Maximum number of steps in the optimization
print time.clock()-t0,"seconds process time"

