import sys
sys.path.insert(0, '../../')

from Task import autochem 
from Basis import sto3g_contracted 

import time

hf = [      (9,(0.0, 0.0,0.12532478)),
      (1,(0.0, 0.0,-1.56968015))]
basis={    9:[    ('S',
    [(63.7352020  ,   0.4301280)]),
    ('S',
    [(11.3448340  ,   0.6789140)]),
    ('S',
    [(2.4985480   ,   0.0494720)]),
    ('S',
    [(0.6336980   ,   0.9637820)]),
    ('P',
    [(2.4985480   ,   0.5115410)]),
    ('P',
    [(0.6336980   ,   0.6128200)])],
    1:[    ('S',
    [(1.309756377 , 0.430128498)]),
    ('S',
    [(0.233135974 , 0.678913531)])],
}

t0 = time.clock()
tasks = ['Energy','Opt','Opt','Opt','Opt']
argnum = [[],[5],[2],[0],[2]]
name = 'hf-sto-2g'
autochem(tasks,        ## Tasks
         hf,          ## Geometry
         sto3g_contracted,
         10/2,          ## Number of electrons
         shifted=False,      ## If the basis is going to be on the atoms coordinates
         name=name,   ## Prefix for all optput files
         verbose=True,       ## If there is going to be an output
         readguess=True,    ## If the optimization will use an initial guess of C
         argnum=argnum,## Order of optimizations ([], for energy)
         maxiter=5)         ## Maximum number of steps in the optimization
print time.clock()-t0,"seconds process time"

