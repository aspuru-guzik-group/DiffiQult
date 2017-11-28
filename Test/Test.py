import unittest
from diffiqult import Task 
from diffiqult.Molecule import System_mol, Getbasis,Getgeom
from diffiqult.Integrals import overlapmatrix, normalization, nuclearmatrix, kineticmatrix, erivector
from diffiqult.Basis import basis_set_3G_STO 
from diffiqult.Tools import getgammp,eri_index, matrix_tovector

import algopy 
from algopy import UTPM, zeros

import numpy as np
from numpy.testing import assert_approx_equal

import scipy
from scipy.special import gammainc,gamma

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
h2o = [(8,(0.0,0.0,0.091685801102911746)),
       (1,(1.4229678834888837,0.0,-0.98120954931681137)),
       (1,(-1.4229678834888837,0.0,-0.98120954931681137))]

basis_h2o =[
[(0,0,0),49.980971,0.430128498,(0.0,0.0,0.091685801102911746)],
[(0,0,0),8.8965880,0.678913531,(0.0,0.0,0.091685801102911746)],
[(0,0,0),1.9452370,0.51154070 ,(0.0,0.0,0.091685801102911746)],
[(0,0,0),0.4933630,0.612819896,(0.0,0.0,0.091685801102911746)],
[(1,0,0),1.9452370,0.51154070,(0.0,0.0,0.091685801102911746)],
[(0,1,0),1.9452370,0.51154070,(0.0,0.0,0.091685801102911746)],
[(0,0,1),1.9452370,0.51154070,(0.0,0.0,0.091685801102911746)],
[(1,0,0),0.4933630,0.612819896,(0.0,0.0,0.091685801102911746)],
[(0,1,0),0.4933630,0.612819896,(0.0,0.0,0.091685801102911746)],
[(0,0,1),0.4933630,0.612819896,(0.0,0.0,0.091685801102911746)],
[(0,0,0),1.309756377,0.430128498,(1.4229678834888837,0.0,-0.98120954931681137)],
[(0,0,0),0.233135974 ,0.678913531,(1.4229678834888837,0.0,-0.98120954931681137)],
[(0,0,0),1.309756377,0.430128498,(-1.4229678834888837,0.0,-0.98120954931681137)],
[(0,0,0),0.233135974 ,0.678913531,(-1.4229678834888837,0.0,-0.98120954931681137)]]

ch4 =[(6,( 0.00000000000000,  0.000000000000000,0.000000000000000)),
      (1,(1.182181057825485, -1.182181057825485,1.182181057825485)),
      (1,(-1.182181057825485, 1.182181057825485,1.182181057825485)),
      (1,(1.182181057825485, 1.182181057825485,-1.182181057825485)),
      (1,(-1.182181057825485, -1.182181057825485,-1.182181057825485))]

hf = [(9,(0.0, 0.0,0.20165898)),
     (1,(0.0, 0.0,-1.64601435))]

hf = [(9,(0.0, 0.0,0.20165898)),
     (1,(0.0, 0.0,-1.24601435))]

h2 = [(1,(0.0,0.0,0.6921792096923653)),
       (1,(0.0,0.0,-0.6921792096923653))]

hcn = [(1,(0.0,0.0,0.0)),
       (6,(0.0, 0.0, 2.0125581778365533)),
       (7,(0.0, 0.0, 4.1914122426680516))]
basis_h2 =[
      [(0,0,0),1.309756377 ,0.430128498, (0.0,0.0,0.6921792096923653)],
      [(0,0,0),0.233135974 ,0.678913531,(0.0,0.0,0.6921792096923653)],
      [(0,0,0),1.309756377 ,0.430128498, (0.0,0.0,-0.6921792096923653)],
      [(0,0,0),0.233135974 ,0.678913531,(0.0,0.0,-0.6921792096923653)]]

ch2o = [      (8,(0.0, 0.0,1.33648844)),
      (6,(0.0, 0.0,-0.97135846)),
      (1,(0.0, 1.76607674,-2.17632039)),
      (1,(0.0, -1.76607674,-2.17632039))]


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

select_geom = {'h2'  : h2,
               'hf'  : hf,
               'h2o' : h2o,
               'ch4' : ch4,
               'ch2o': ch2o,
               'hcn' : hcn}
select_e = {'h2'  : 2,
            'hf'  : 10,
            'h2o' : 10,
            'ch4' : 10,
            'ch2o': 16,
            'hcn' : 14}
select_energy = {'h2': -1.09648608604,
               'hf' : -72.9771864477,
               'h2o' : -72.9771864477,
               'ch2o': -112.35254394,
               'ch4' : -38.6835836392,
               'hcn' : -82.6357075703}
select_basis = {'h2' : basis_set,
               'hf'  : basis_set,
               'h2o' : basis_set,
               'ch4' : basis_set,
               'ch2o': basis_set_3G_STO,
               'hcn' : basis_set}
test_mol_names = ['h2','h2o','ch4','hcn','ch2o']



class System_test(System_mol):
    def __init__(self,mol,shifted=False):
          self.mol = select_geom.get(mol)
          System_mol.__init__(self,self.mol,select_basis.get(mol),select_e.get(mol),mol)
          self.tape = "./test/"+mol
          self.energy = select_energy.get(mol)
          self.coef = normalization(np.array(self.alpha),self.coef,self.l,self.list_contr)
          #coef = normalization(np.array(self.alpha),np.copy(self.coef),self.xyz,self.l,self.list_contr)
          self.alpha_algopy = UTPM.init_jacobian(self.alpha)
          self.coef_algopy = UTPM.init_jacobian(self.coef)
          return 
         
        

'''Test classes'''
class Test_gamma(unittest.TestCase):
    def setUp(self):
        pass
        #return

    def test_gamma(self):
         '''This module test the function getgammp'''
         print "Test gamma incomplete ... start"
         for i in range(0,10):
             m = i + 0.5
             for t in np.arange(1.0e-12,9,0.01): 
                ## Please note that the function has a by pass
                ## before this function to avoid ceros.
                gammafvalue = getgammp(m,t)
                self.assertAlmostEqual(gammafvalue,gammainc(m,t)*gamma(m), msg='Error in gamma incomplete',places=5)
         print "Test gamma incomplete ... done"
         #return
             
        

class Test_Molecules(unittest.TestCase):
    def setUp(self):
        self.Molecules = []
        for mol in test_mol_names:
            print mol
            self.Molecules.append(System_test(mol))
        pass


    def test_oneeintegrals(self):
        functions = [
                     self.__test_overlapmatrix__,
                     self.__test_kineticmatrix__,
                     self.__test_nuclearmatrix__]
        function_names = ['Overlap matrix','Kinetic matrix','Nuclear matrix']
        for i, func in enumerate(functions):
           print ('Testing '+function_names[i]+'...start')
           for Mol in self.Molecules:
               print ('    Testing '+Mol.mol_name+'...start')
               func(Mol)
               print ('    Testing '+Mol.mol_name+'...done')
           print ('Testing '+function_names[i]+'...done')
        pass

    def test_twointegrals(self):
        print ('Testing Eris'+'...start')
        for Mol in self.Molecules:
            print ('    Testing '+Mol.mol_name+'...start')
            self.__test_eris__(Mol)
            print ('    Testing '+Mol.mol_name+'...done')
        print ('Testing Eris'+'...done')
        pass


    def __test_overlapmatrix__(self,Mol):
        ''' In computes the overlap matrix 
        and compare it with the file benckmark '''
         
        tool = 1E-7
        epsilon = 1e-6
  
        S  = overlapmatrix(Mol.alpha,Mol.coef,Mol.xyz,Mol.l,Mol.nbasis,Mol.list_contr,np.float64(1.0))
        f = open(Mol.tape+'_overlap_pyquante.out' ,'r')
        f_lines = f.read().split('\n')
        line = 0
        for i in range(Mol.nbasis):
          for j in range(Mol.nbasis):
              self.assertAlmostEqual(S[i,j],float(f_lines[line].split()[2]),msg="Error: Test Overlap "+Mol.mol_name,places=5)
              line += 1
        f.close()


        grad_algo_alpha = UTPM.extract_jacobian(overlapmatrix(Mol.alpha_algopy,Mol.coef,Mol.xyz,Mol.l,Mol.nbasis,Mol.list_contr,Mol.alpha_algopy))
        ### Testing grad alpha:
        for i in range(len(Mol.alpha)):
          for j in range(len(Mol.alpha)):
              alpha_epsilon = np.copy(Mol.alpha)
              alpha_epsilon[i] = Mol.alpha[i]+ epsilon
              Sij_epsilon = overlapmatrix(alpha_epsilon,Mol.coef,Mol.xyz,Mol.l,Mol.nbasis,Mol.list_contr,np.float64(1.0))
              dSij_da = (Sij_epsilon- S)/epsilon
              np.testing.assert_almost_equal(dSij_da,grad_algo_alpha[:,:,i],decimal=3,verbose=True,err_msg='Error: Test Overlap Grad')

        grad_algo_coef = UTPM.extract_jacobian(overlapmatrix(Mol.alpha,Mol.coef_algopy,Mol.xyz,Mol.l,Mol.nbasis,Mol.list_contr,Mol.alpha_algopy))
        ### Testing grad alpha:
        for i in range(len(Mol.coef)):
          for j in range(len(Mol.coef)):
              coef_epsilon = np.copy(Mol.coef)
              coef_epsilon[i] = Mol.coef[i]+ epsilon
              Sij_epsilon = overlapmatrix(Mol.alpha,coef_epsilon,Mol.xyz,Mol.l,Mol.nbasis,Mol.list_contr,np.float64(1.0))
              dSij_da = (Sij_epsilon- S)/epsilon
              np.testing.assert_almost_equal(dSij_da,grad_algo_coef[:,:,i],decimal=3,verbose=True,err_msg='Error: Test Overlap Grad')
        pass
  
    def __test_kineticmatrix__(self,Mol):
        ''' In computes the kinetics matrix 
        and compare it with the file benckmark '''
         
        tool = 1E-7
        epsilon = 1e-5
  
        T  = kineticmatrix(Mol.alpha,Mol.coef,Mol.xyz,Mol.l,Mol.nbasis,Mol.list_contr,np.float64(1.0))

        f = open(Mol.tape+'_kinetics_pyquante.out' ,'r')
        f_lines = f.read().split('\n')
        line = 0
        for i in range(Mol.nbasis):
          for j in range(Mol.nbasis):
              self.assertAlmostEqual(T[i,j],float(f_lines[line].split()[2]),msg="Error: Test Kinetic "+Mol.mol_name,places=5)
              line += 1
        f.close()
         

        grad_algo_alpha = UTPM.extract_jacobian(kineticmatrix(Mol.alpha_algopy,Mol.coef,Mol.xyz,Mol.l,Mol.nbasis,Mol.list_contr,Mol.alpha_algopy))
        ### Testing grad alpha:
        for i in range(len(Mol.alpha)):
          for j in range(len(Mol.alpha)):
              alpha_epsilon = np.copy(Mol.alpha)
              alpha_epsilon[i] = Mol.alpha[i]+ epsilon
              Tij_epsilon  = kineticmatrix(alpha_epsilon,Mol.coef,Mol.xyz,Mol.l,Mol.nbasis,Mol.list_contr,np.float64(1.0))
              dTij_da = (Tij_epsilon- T)/epsilon
              np.testing.assert_almost_equal(dTij_da,grad_algo_alpha[:,:,i],decimal=3,verbose=True,err_msg='Error: Test Overlap Grad')

        grad_algo_coef = UTPM.extract_jacobian(kineticmatrix(Mol.alpha,Mol.coef_algopy,Mol.xyz,Mol.l,Mol.nbasis,Mol.list_contr,Mol.coef_algopy))
        for i in range(len(Mol.alpha)):
          for j in range(len(Mol.alpha)):
              coef_epsilon = np.copy(Mol.coef)
              coef_epsilon[i] = Mol.coef[i]+ epsilon
              Tij_epsilon  = kineticmatrix(Mol.alpha,coef_epsilon,Mol.xyz,Mol.l,Mol.nbasis,Mol.list_contr,np.float64(1.0))
              dTij_da = (Tij_epsilon- T)/epsilon
              np.testing.assert_almost_equal(dTij_da,grad_algo_coef[:,:,i],decimal=3,verbose=True,err_msg='Error: Test Overlap Grad')
        pass


    def __test_nuclearmatrix__(self,Mol):
        ''' In computes the kinetics matrix 
        and compare it with the file benckmark '''
         
        epsilon = 1e-6
  
        V  = nuclearmatrix(Mol.alpha,Mol.coef,Mol.xyz,Mol.l,Mol.nbasis,Mol.charges,Mol.atom,Mol.natoms,Mol.list_contr,np.float64(1.0))
        f = open(Mol.tape+'_nuclear_pyquante.out' ,'r')
        f_lines = f.read().split('\n')
        line = 0
        for i in range(Mol.nbasis):
          for j in range(Mol.nbasis):
              self.assertAlmostEqual(V[i,j],float(f_lines[line].split()[2]),msg="Error: Test Nuclear"+Mol.mol_name,places=5)
              line += 1
        f.close()

        ### Testing grad alpha:
        grad_algo_alpha = UTPM.extract_jacobian(nuclearmatrix(Mol.alpha_algopy,Mol.coef,Mol.xyz,Mol.l,Mol.nbasis,
                                                              Mol.charges,Mol.atom,Mol.natoms,Mol.list_contr,Mol.alpha_algopy))

        for i in range(len(Mol.alpha)):
          for j in range(len(Mol.alpha)):
              alpha_epsilon = np.copy(Mol.alpha)
              alpha_epsilon[i] = Mol.alpha[i]+ epsilon
              Vij_epsilon  = nuclearmatrix(alpha_epsilon,Mol.coef,Mol.xyz,Mol.l,Mol.nbasis,Mol.charges,Mol.atom,Mol.natoms,Mol.list_contr,np.float64(1.0))
              dVij_da = (Vij_epsilon- V)/epsilon
              np.testing.assert_almost_equal(dVij_da,grad_algo_alpha[:,:,i],decimal=3,verbose=True,err_msg='Error: Test Nuclear Grad')

        grad_algo_coef = UTPM.extract_jacobian(nuclearmatrix(Mol.alpha,Mol.coef_algopy,Mol.xyz,Mol.l,Mol.nbasis,
                                                              Mol.charges,Mol.atom,Mol.natoms,Mol.list_contr,Mol.alpha_algopy))
        for i in range(len(Mol.alpha)):
          for j in range(len(Mol.alpha)):
              coef_epsilon = np.copy(Mol.coef)
              coef_epsilon[i] = Mol.coef[i]+ epsilon
              Vij_epsilon  = nuclearmatrix(Mol.alpha,coef_epsilon,Mol.xyz,Mol.l,Mol.nbasis,Mol.charges,Mol.atom,Mol.natoms,Mol.list_contr,np.float64(1.0))
              dVij_da = (Vij_epsilon- V)/epsilon
              np.testing.assert_almost_equal(dVij_da,grad_algo_coef[:,:,i],decimal=3,verbose=True,err_msg='Error: Test Nuclear Grad')
        pass


    def __test_eris__(self,Mol,grad=False):
        ''' In computes the Eris tensor 
        and compare it with the file benckmark '''

        epsilon = 1e-6
        nbasis = Mol.nbasis

        Eris = erivector(Mol.alpha,Mol.coef,Mol.xyz,Mol.l,Mol.nbasis,Mol.list_contr,np.float(1.0))
    
        f = open(Mol.tape+'_eri_pyquante.out' ,'r')
        f_lines = f.read().split('\n')
        for line in f_lines[:-1]:
            j,k,o,m,eri_pyquante = line.split()
            #print j,k,o,m, Eris[eri_index(int(j),int(k),int(o),int(m),nbasis)],float(eri_pyquante)
            self.assertAlmostEqual(Eris[eri_index(int(j),int(k),int(o),int(m),nbasis)],float(eri_pyquante),msg="Error: Test Eris "+Mol.mol_name,places=5)
        f.close()
        
        pass
        if grad:
           self.test_erisgrad__(Mol,Eris)
        pass

    def __test_erisgrad__(self,Mol,Eris):
       ''' In computes the Eris tensor 
       and compare it with the file benckmark '''

       epsilon = 1e-6

       grad_algo = UTPM,extract_jacobian(erivector(Mol.alpha_algo,Mol.coef,Mol.xyz,Mol.l,Mol.nbasis,Mol.alpha_algo))
    
       for i in range(self.nbasis):
          alpha_epsilon = self.alpha[i] + epsilon
          alpha_i = self.alpha[i]
          for j in range(self.nbasis):
             alpha_j = self.alpha[j]
             for k in range(self.nbasis):
                alpha_k = self.alpha[k]
                for m in range(self.nbasis):
                   fact = 1
                   Eriij_epsilon = eriss(alpha_epsilon,self.coef[i],self.xyz[i],self.l[i],alpha_j,self.coef[j],self.xyz[j],self.l[j],
                                       alpha_k,self.coef[k],self.xyz[k],self.l[k],self.alpha[m],self.coef[m],self.xyz[m],self.l[m])
                   dEriij_da = (Eriij_epsilon- Eris[eri_index(i,j,k,m,self.nbasis)])/epsilon
                   if i == j:
                      dEriij_da *= 2.0
                   if i == m:
                      Eriij_epsilon = eriss(alpha_i,self.coef[i],self.xyz[i],self.l[i],alpha_j,coef[j],xyz[j],l[j],
                                            alpha_k,self.coef[k],self.xyz[k],self.l[k],alpha_epsilon,self.coef[m],self.xyz[m],self.l[m])
                      dEriij_da += (Eriij_epsilon- Eris[eri_index(i,j,k,m,self.nbasis)])/epsilon
                   if i == k:
                      Eriij_epsilon = eriss(alpha_i,self.coef[i],self.xyz[i],self.l[i],alpha_j,self.coef[j],self.xyz[j],self.l[j],
                                            alpha_epsilon,self.coef[k],self.xyz[k],self.l[k],self.alpha[m],self.coef[m],self.xyz[m],self.l[m])
                      dEriij_da += (Eriij_epsilon- Eris[eri_index(i,j,k,m,self.nbasis)])/epsilon
                   self.assertAlmostEqual(dEriij_da,grad_algo[eri_index(i,j,k,m,nbasis),i], msg="Error: Test Eris grad"+Mol.mol_name,places=6)
                   #diff = dEriij_da - grad_algo[eri_index(i,j,k,m,nbasis),i]
                   #if abs(diff) > tool:
                   #    print dEriij_da, grad_algo[eri_index(i,j,k,m,nbasis),i]
                   #    print i,j,k,m
                   #    print "Error: Test Eri Grad"
                   #    exit()
       pass

    def energy(self):
        print ('Testing Energy'+'...start')
        tasks = ['Energy']
        argnum = [[]]
        for Mol in self.Molecules:
            print ('    Testing '+Mol.mol_name+'...done')
            energy = self._test_runpoint(Mol,tasks,argnum)
            self.assertAlmostEqual(select_energy.get(Mol.mol_name),energy,msg="Error: Test Energy "+Mol.mol_name,places=6)
            self._test_energy_grad(Mol)
        print ('Testing Energy'+'...done')
        pass

    def _test_runpoint(self,Mol,tasks,argnum=[0]):
        sys_mol = autochem(tasks,        ## Tasks
                              Mol.mol,          ## Geometry
                              select_basis.get(Mol.mol_name),   ## Basis set (if shifted it should have the coordinates too)
                              Mol.ne/2,          ## Number of electrons
                              shifted=False,      ## If the basis is going to be on the atoms coordinates
                              name=Mol.mol_name,   ## Prefix for all optput files
                              verbose=True,       ## If there is going to be an output
                              argnum=argnum)## Order of optimizations ([], for energy)
        return Mol.energy
  
    def _test_energy_grad(self,Mol):
        epsilon = 1e-5
        current_e = Mol.energy
        tasks = ['Grad']
        ### Alphas
        argnum = [0]
        fake_e = self._test_runpoint(Mol,tasks,argnum=argnum)
        grad = Mol.grad
        for i in range(len(grad_mol.alpha)):
            grad_mol = Mol
            grad_mol.alpha[i] = grad_mol.alpha[i]
            energy_epsilon = self._test_runpoint(grad_mol,tasks,argnum)
            dE = (energy-energy_epsilon)/epsilon
        pass

    def _test_bfgs(self):
        from diffiqult.Optimize import rosen,rosen_der,algopy_wrapper,fmin_bfgs

      
        print ('    Testing optimizer ...start')
        x0 = np.array([0.8, 1.2, 0.7])
        x_not = np.array([1.0,1.0,1.0])
        x1 = fmin_bfgs(rosen, x0, fprime=rosen_der,maxiter=80)
        self.assertAlmostEqual(x_not,x1,msg="Error: Test optimizer with rosen ",places=6)

        x1 = fmin_bfgs(rosen, x0, fprime=algopy_wrapper(rosen),gtol=1e-4, maxiter=100)
        self.assertAlmostEqual(x_not,x1,msg="Error: Test optimizer with rosen and algopy",places=6)
        print ('    Testing optimizer ...done')

        pass

        
    def test_optimizer(self):
        '''This function test the optmization, there is not bechmark yet :S
        for the moment there is just a way to check if is minimizes energy and
        if it doesn't break 
        the functions are taken from the module Wrap'''
        print ('Testing Optimization'+'...start')
        self._test_bfgs()
        for mol in self.Molecules[0]:
            print ('    Testing '+Mol.mol_name+'...start')
            manager = Tasks(mol,
                     name='../testfiles/%s'%(Mol.mol_name),      ## Prefix for all optput files
                     verbose=False)          ## If there is going to be an output
            manager.runtask('Opt',
                     max_scf=50,
                     max_steps = 5,
                     print_coef=False,
                     argnum=[2],
                     output=False)

            print ('    Testing '+Mol.mol_name+'...done')
        print ('Testing Optimization'+'...done')
        pass
   
if __name__ == '__main__':
    unittest.main(module='Test')
