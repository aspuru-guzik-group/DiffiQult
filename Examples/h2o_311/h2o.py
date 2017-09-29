import sys
sys.path.insert(0, '../../')
import Task
from Task  import autochem
import time

h2o = [      (8,(0.0, 0.0, 0.091685801102911746)),
      (1,(1.4229678834888837, 0.0, -0.98120954931681137)),
      (1,(-1.4229678834888837, 0.0, -0.98120954931681137))]
basis={    8:[    ('S',
    [(322.0370000 ,   0.0592394),
    (48.4308000  ,   0.3515000),
    (10.4206000  ,   0.7076580)])
      ,    ('S',
    [(7.4029400   ,  -0.4044530),
    (1.5762000   ,   1.2215600)])
      ,    ('S',
    [(0.3736840   ,   1.0000000)])
      ,    ('P',
    [(7.4029400   ,   0.2445860),
    (1.5762000   ,   0.8539550)])
      ,    ('P',
    [(0.3736840   ,   1.0000000)])
      ],
    1:[    ('S',
    [(5.4471780   ,   0.1562850),
    (0.8245470   ,   0.9046910)])
      ,    ('S',
    [(0.1831920   ,   1.0000000)])
      ]}

t0 = time.clock()
tasks = ['Energy','Opt','Opt','Opt','Opt']
argnum = [[],[2],[0,1],[2],[0,1],[2]]
name = 'h2o'
ne = 10
autochem(tasks,        ## Tasks
         h2o,          ## Geometry
         basis,   ## Basis set (if shifted it should have the coordinates too)
         ne/2,          ## Number of electrons
         shifted=False,      ## If the basis is going to be on the atoms coordinates
         name=name,   ## Prefix for all optput files
         verbose=True,       ## If there is going to be an output
         readguess=True,    ## If the optimization will use an initial guess of C
         argnum=argnum,## Order of optimizations ([], for energy)
         maxiter=10)         ## Maximum number of steps in the optimization
print time.clock()-t0,"seconds process time"
