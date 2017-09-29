import sys
import time 
sys.path.insert(0, '../../')

from Task import autochem 
 
d = -1.64601435
mol = [(1,(0.0,0.0,0.20165898)),
(1,(0.0,0.0,d))]


basis = {1: [('S',
 [(  35.52322122,           0.00916359628)]),
   ('S',
 [(  6.513143725,           0.04936149294)]),
   ('S',
 [(  1.822142904,           0.16853830490)]),
   ('S',
 [(  0.625955266,           0.37056279970)]),
   ('S',
 [(  0.243076747,           0.41649152980)]),
   ('S',
 [(  0.100112428,           0.13033408410)])]}

tasks = ["Grad"]
argnum = [[0]]
autochem(tasks,        ## Tasks
         mol,          ## Geometry
         basis,   ## Basis set (if shifted it should have the coordinates too)
         2/2,          ## Number of electrons
         shifted=False,      ## If the basis is going to be on the atoms coordinates 
         name='h2_sto_3g',   ## Prefix for all optput files
         verbose=True,       ## If there is going to be an output
         argnum=argnum,## Order of optimizations ([], for energy)
         maxiter=5)         ## Maximum number of steps in the optimization
