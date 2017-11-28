"""
=================================================================
TODO
  DiffiQult: an autodifferentiable electronic structure package
==================================================================
Description:
Rationale:
How to cite DiffiQulT::
    
"""

import os
__install_path__ = os.path.realpath(__file__)


# check that dependencies are satisfied


_min_numpy_version = '1.5.0'
_preferred_numpy_version = '1.6.2'
_min_scipy_version = '0.11.0'

try:
    import numpy
    from numpy.lib import NumpyVersion

    # ignore warnings "ComplexWarning: Casting complex values to real discards the imaginary part"
    import warnings
    warnings.simplefilter("ignore", numpy.ComplexWarning)

except ImportError as e:
    raise ImportError(
            "NumPy import error (%s)\n"
            "Please install NumPy >= %s" % (e, _preferred_numpy_version))

if NumpyVersion(numpy.version.version) < _min_numpy_version:
    raise ImportError(
            "NumPy version %s was detected.\n"
            "Please install NumPy >= %s" % (
                numpy.version.version, _preferred_numpy_version))

try:
    import scipy
except ImportError as e:
    raise ImportError(
        "SciPy import error (%s)\n"
        "Please install SciPy >= " + _min_scipy_version)

if NumpyVersion(scipy.version.version) < _min_scipy_version:
    raise ImportError(
            "SciPy version %s was detected.\n"
            "Please install SciPy >= %s" % (
                scipy.version.version, _min_scipy_version))


try:
   import algopy
except ImportError as e:
    raise ImportError(
        "algopy import error (%s)\n"
        "algopy is a requirement of DiffiQulT.\n"
        "Please install algopy")

# testing
from numpy.testing import Tester
test = Tester().test

# import standard submodules and important classes/functions
from Task import Tasks
from Molecule import System_mol

