import unittest
import Task
from Task import getbasis, getgeom
import Integrals
from Integrals import normalization, eriss
import numpy as np
from numpy.testing import assert_approx_equal
import scipy
from scipy.special import gammainc,gamma
import Tools
from Tools import getgammp
import unittest

'''
This module contains test functions
for the elements and gradients of integrals
of s-type and p-type orbitals, and the results are compared 
with py-quante outputs and numerical derivatives.
'''
### TODO:
### Write unittest for gradients of xyz

### Some global variables: geometry, and basis sets, and basis+geometry for
### shifted geometries
### We need to do a sophisticated way to read basis and xyz from a file
hcn = [(1,(0.0,0.0,0.0)),
       (6,(0.0, 0.0, 2.0125581778365533)),
       (7,(0.0, 0.0, 4.1914122426680516))]

basis_set = \
{1: [('S',
     [(1.309756377 ,0.430128498)]),     
     ('S',
      [(0.233135974 ,0.678913531)])], 
 6: [('S',
       [(27.38503303,0.430128498)]),
     ('S',
       [(4.874522052,0.678913531)]),
     ('S',
       [(1.136748189,0.51154070)]),
     ('S',
       [(0.288309360,0.612819896)]),
     ('P',
       [(1.136748189,0.51154070)]),
     ('P',
       [(0.288309360,0.612819896)])],
 7: [('S',
       [(99.106168999999994, 0.15432897000000001)]),
     ('S',
       [(18.052312000000001, 0.53532813999999995)]),
     ('S',
       [(4.8856602000000002, 0.44463454000000002)]),
     ('S',
       [(3.7804559000000002, -0.099967230000000004)]),
     ('S',
       [(0.87849659999999996, 0.39951282999999999)]),
     ('S',
       [(0.28571439999999998, 0.70011546999999996)]),
     ('P',
      [(3.7804559000000002, 0.15591627)]),
     ('P',
      [(0.87849659999999996, 0.60768372000000004)]),
     ('P',
      [(0.28571439999999998, 0.39195739000000002)])],
 8: [('S',
      [(49.9809710, 0.4301280)]),
     ('S',
       [(8.8965880, 0.6789140)]),
     ('S',
       [(1.9452370,0.0494720)]),
     ('S',
       [(0.4933630 ,0.9637820)]),
     ('P',
       [(1.9452370,0.0494720)]),
     ('P',
       [(0.4933630 ,0.9637820)])],
 9:[ ('S',
       [(63.7352020  ,   0.4301280)]),
     ('S',
       [(11.3448340  ,   0.6789140)]),
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
}

select_geom = {'hcn' : hcn}
select_e = {'hcn' : 14}
select_energy = {'hcn' : -82.6357075703}
test_mol_names = ['hcn']


class System_mol():
    '''This class contains all the information of the system
    extracted from mol and basis'''
    
    def __init__(self,mol,basis,ne,mol_name):

          self.mol_name = mol_name
          ## Info for basis
          Basis = getbasis(mol,basis_set)                                  # Get basis
          self.nbasis = len(Basis.alpha)
          self.alpha = np.array(Basis.alpha)                                    # Alpha
          #self.alpha_algopy = UTPM.init_jacobian(Basis.alpha)
          self.xyz = np.reshape(np.array(Basis.xyz,dtype='float64'),(self.nbasis,3)) # Nuclear coordinates
          self.l = Basis.l          
 
          ## Geometry
          Geom = getgeom(mol)                           # Get basis
          self.charges = np.array(Geom.charge)               # Charges
          self.atom = np.array(Geom.xyz)                     # Alpha
          self.natoms = len(self.charges)
    
          ## Normalization
          self.coef = np.sign(Basis.coef)*normalization(np.array(Basis.alpha),self.xyz,self.l,self.nbasis)
   
          ## Number of electrons
          self.ne = ne
          return

class System_test(System_mol):
    def __init__(self,mol):
          self.mol = select_geom.get(mol)
          System_mol.__init__(self,self.mol,basis_set,select_e.get(mol),mol)
          self.tape = "./test/"+mol
          self.energy = select_energy.get(mol)
          return 
         
        


class Test_Molecules(unittest.TestCase):
    def setUp(self):
        self.Molecules = []
        for mol in test_mol_names:
            self.Molecules.append(System_test(mol))
        pass


    def test_twowintegrals(self):
        print ('Testing Eris'+'...start')
        for Mol in self.Molecules:
            print ('    Testing '+Mol.mol_name+'...start')
            print 'This case works'
            self.__test_eris_elem__(Mol)
            print 'This case does not work'
            self.__test_eris_elem__(Mol, i=6,j=8,k=20,m=6)
            print ('    Testing '+Mol.mol_name+'...done')
        print ('Testing Eris'+'...done')
        pass



    def __test_eris_elem__(self,Mol,i=6,j=8,k=6,m=20):
        ''' In computes the Eris tensor 
        and compare it with the file benckmark '''

        nbasis = Mol.nbasis

        eri = eriss(Mol.alpha[i],Mol.coef[i],Mol.xyz[i],Mol.l[i],Mol.alpha[j],Mol.coef[j],Mol.xyz[j],Mol.l[j],
                        Mol.alpha[k],Mol.coef[k],Mol.xyz[k],Mol.l[k],Mol.alpha[m],Mol.coef[m],Mol.xyz[m],Mol.l[m])
        print 'eri',eri
        pass

if __name__ == '__main__':
    unittest.main(module='Test')
