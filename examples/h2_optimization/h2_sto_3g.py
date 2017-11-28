from diffiqult.Molecule import System_mol
from diffiqult.Task import Tasks
from diffiqult.Basis import basis_set_3G_STO as basis
 
d = -1.64601435
mol = [(1,(0.0,0.0,0.20165898)),
(1,(0.0,0.0,d))]
ne = 2


system = System_mol(mol,                                ## Geometry
                    basis,                              ## Basis set (if shifted it should have the coordinates too)
                    ne,                                 ## Number of electrons
                    shifted=False,                      ## If the basis is going to be on the atoms coordinates 
                    angs=False,                         ## Units -> Bohr
                    mol_name='agua')                    ## Units -> Bohr

manager = Tasks(system,
                name='h2_sto_3g',      ## Prefix for all optput files
                verbose=True)          ## If there is going to be an output


manager.runtask('Opt',
                max_scf=50,
                printcoef=False,
                argnum=[0,1],
                output=True)

manager.end()

