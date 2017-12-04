.. _intro:

DiffiQult
===========

*DiffiQult* is an open source autodifferentiable quantum chemistry package.


.. only:: html

   .. figure:: h2o_3g_MO_1.gif

Method:

* RHF 

Features:

* Single point calculations
* Energy gradients with respect to any parameter of the one-particle basis functions.
* Energy optimization with respect of any parameter of the Gaussian basis functions.

===============
Getting started with DiffiQult
===============

Requirements
______________________________

* Numpy

* Algopy
 	* Official releases and installation:

		Available at: http://pypi.python.org/pypi/algopy

		``pip install algopy``

* Python 2.7 (so far tested). 

Installation
______________________________

* From source:

     ``git clone https://github.com/ttamayo/DiffiQult.git``

     ``python setup.py install``


===============
Usage
===============


Molecular system 
______________________________

We define the parameters of a molecular systems with an ``System_mol`` object:

* molecular geometry in xyz format and atomic units
* basis sets (data base so far sto_3G
* number of electrons

For example:

.. code-block:: python

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

Tasks
______________________________

The jobs in *Diffiqult* are managed by a ``Tasks`` object,


.. code-block:: python

   manager = Tasks(system,
                name='h2_sto_3g',      ## Prefix for all optput files
                verbose=True)          ## If there is going to be an output

where we defined the molecular system to 
optimize with the object ``system``, and output options with ``verbose``. 

The class ``Task`` contains the method ``Tasks.runtask``, it computes one the following options: 

+----------------------+--------------+-------------------------------------------------------------------+
| Task                 | Key          | Description                                                       |
+======================+==============+===================================================================+
| Single point energies| ``Energy``   | It calculates the RHF energy and updates some attibute in system  |
+----------------------+--------------+-------------------------------------------------------------------+
| Optimization         | ``Opt``      | It optimizes a given parameter and updates the basis set in system|
+----------------------+--------------+-------------------------------------------------------------------+


Single point calculation
`````````````

.. code-block:: python

        manager.runtask('Energy',
                     max_scf=50,                        # Maximum number of SCF cycles
                     printcoef=True,                    # This will produce a npy file with the molecular coefficients
                     name='Output.molden',              # Name of the output file (Compatible with molden)
                     output=True)

**Notes:** 

* We currently don't have convergence options for the SCF.
* The molden file also contains an input section that can be used as input for system with the option ``shifted``
* The geometry and MOs can be vizualized with *molden*,and the molden file.

Optimization
`````````````

To optimize one or many input parameters, we use the option ``Opt``. After a succesful optimization or
If the optimization reaches the maximum number of steps or convergence, it updates
the attributes of the ``system_mol`` object.

.. code-block:: python

    manager.runtask('Opt',
                     max_scf=50,
                     printcoef=False,
                     argnum=[2],  # Optimization of centers
                     output=True) # We optimized all the steps
    print(manager.syste.energy)

where ``argnum`` recieves a list with the parameters to optimize with the following convention:

+--------------------------+------------+
| Parameter                | ``argnum`` |
+==========================+============+
| Widths                   |  0         | 
+--------------------------+------------+
| Contraction coefficients |  1         |
+--------------------------+------------+
| Gaussian centers         |  2         |
+--------------------------+------------+

for example, we can optimize the atomic centered basis function with respect of their widths and contraction
coefficients in the following way.


.. code-block:: python

 manager.runtask('Opt',
                     max_scf=50,
                     printcoef=False,
                     argnum=[0,1],  # Optimization of centers
                     output=True)   # We print a molden file of all steps

Additionally, if ``output`` is set to ``True``, a molden file of each optimization step is printed.
