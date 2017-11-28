# DiffiQult

A fully autodifferentiable Hatree-Fock algorithm capable of optimizing any parameter of the basis set.


*DiffiQult* optimizes the RHF energy with respect to widths, contraction coefficients and Gaussian centers. 

# Requirements

* Numpy
* Algopy
 	* Official releases and installation:

		Available at: http://pypi.python.org/pypi/algopy

		```pip install algopy```

* Python 2.7 (tested so far). 

# Installation

* From source:

     ```git clone https://github.com/ttamayo/DiffiQult.git```

     ```python setup.py install```

* From pip:

``` pip install diffiqult```

# Usage

## Molecular system 

We define the molecular systems by creating an object of the class  ```System_mol```, where we specify the
molecular geometry in atomic units, basis sets and number of electrons, optionally we can 
specify a tag. For example:

```
# Basis set is sto_3G
from diffiqult.Basis import basis_set_3G_STO as basis
     # Our molecule H_2
     d = -1.64601435
     mol = [(1,(0.0,0.0,0.20165898)),(1,(0.0,0.0,d))]
     
     # Number of electrons
     ne = 2
     system = System_mol(mol,                                ## Geometry
                         basis,                              ## Basis set (if shifted it should have the coordinates too)
                         ne,                                 ## Number of electrons
                         shifted=False,                      ## If the basis is going to be on the atoms coordinates 
                         mol_name='agua')                    ## Units -> Bohr
```

## Tasks

The jobs in *Diffiqult* are managed by an Tasks object, where we define the molecular system to 
optimize and output options. For examplem, if we want to obtain a file with the results.

For example:
```
manager = Tasks(system,
                name='h2_sto_3g',      ## Prefix for all optput files
                verbose=True)          ## If there is going to be an output
```

The method ```Tasks.runtask``` computes one the following tasks: 

| Task          | Key         | Description  |
| ------------- |:-------------:| -----:|
| Single point energies| ```Energy``` | Calculate the RHF energy and update it in an attibute in system |
| Optimization  | ```Opt``` |   Optimize a given parameter and update the basis set in sytem|


## Single point calculation

To calculate the RHF of a molecule we use the option ``` `Energy` ```, optionally we can produce a molden file 
with the information of the geometry, basis and MO.

```     manager.runtask('Energy',
                     max_scf=50,                        # Maximum number of SCF cycles
                     printcoef=True,                    # This will produce a npy file with the molecular coefficients
                     name='Output.molden',              # Name of the output
                     output=True)
```

**Notes:** 

* We currently don't have convergence options for the SCF  .
* The molden file also contains an input section that can be used as input for system with the option ```shifted```
* The geometry and MOs can be vizualized with *molden*,and the molden file.

## Optimization

To optimize one or many input parameters, we use the option ````Opt` ```,
it updates the attributes of the molecular system object.

For example:

```
 manager.runtask('Opt',
                     max_scf=50,
                     printcoef=False,
                     argnum=[2],  # Optimization of centers
                     output=True) # We optimized all the steps
```

where ```argnum``` recives a list with the parameters to optimize with the following convention:

| Parameter          | argnum    |
| ------------- |:-------------:|
| Width | 0 | 
| Contraction coefficients | 1 |
| Gaussian centers | 2 |

For example, we can optimize the atomic centered basis function with respect of their widths and contraction
coefficients in the following way.

```
 manager.runtask('Opt',
                     max_scf=50,
                     printcoef=False,
                     argnum=[0,1],  # Optimization of centers
                     output=True)   # We print a molden file of all steps
```

