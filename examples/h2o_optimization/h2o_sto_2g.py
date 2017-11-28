from diffiqult import Tasks
from diffiqult import System_mol
from diffiqult.Basis import basis_set_3G_STO
 
mol = [
( 8,(0.0, 0.0, 0.091685801102911746)),
( 1,(1.4229678834888837, 0.0, -0.98120954931681137)),
( 1,(-1.4229678834888837, 0.0, -0.98120954931681137))]
basis = basis_set_3G_STO 
ne = 10

system = System_mol(mol,                      ## Geometry
                    basis,                    ## Basis set (if shifted it should have the coordinates too)
                    ne,                       ## Number of electrons
                    shifted=False,            ## If the basis is going to be on the atoms coordinates 
                    angs=False,               ## Units -> Bohr
                    mol_name='agua')          ## Units -> Bohr

manager = Tasks(system,
                name='h2_sto_3g',      ## Prefix for all optput files
                verbose=True)          ## If there is going to be an output


manager.runtask('Energy',
                     max_scf=50,
                     printcoef=True,
                     name='Output.molden',
                     output=False)

manager.runtask('Opt',
                     max_scf=50,
                     printcoef=False,
                     maxiter=3,
                     argnum=[0],
                     output=False)

