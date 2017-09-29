from Integrals import nuclearmatrix
from Data import select_weight

import algopy
from algopy import UTPM
import numpy as np

def nuclear_gradient(mol,center = np.array([0.0,0.0,0.0])):
        ''' In computes the geometric derivatives of the kinetics matrix '''
        '''
        grad_algo_xyz = np.zeros((mol.nbasis,mol.nbasis,3))
        #for i in range(len(mol.alpha)):
        for i in range(1):
            xyz = np.array(mol.xyz[i] - center)
            print xyz
            xyz_algopy = UTPM.init_jacobian(xyz)
            charges = [1]
            ncharges = 1
            grad_algo_xyz = grad_algo_xyz + UTPM.extract_jacobian(nuclearmatrix(mol.alpha,mol.coef,mol.xyz,mol.l,mol.nbasis,charges,xyz_algopy,ncharges,xyz_algopy))
        '''
         
        xyz = np.array(center)
        xyz_algopy = UTPM.init_jacobian(xyz)
        charges = [1]
        ncharges = 1
        grad_algo_xyz = UTPM.extract_jacobian(nuclearmatrix(mol.alpha,mol.coef,mol.xyz,mol.l,mol.nbasis,charges,xyz_algopy,ncharges,xyz_algopy))
        print mol.atom
        print grad_algo_xyz[:,:,0]
        print grad_algo_xyz[:,:,1]
        print grad_algo_xyz[:,:,2]
        print grad_algo_xyz.shape 
        
        return grad_algo_xyz

def electronic_contribution(mol,coef_file,center=np.array([[0.0,0.0,0.0]])):
        '''This calculates dde electronic contribution of mu centered at center'''
        print center
        nbasis = mol.nbasis
        ne = mol.ne
	C = np.load(coef_file)
        D = np.zeros((nbasis,nbasis))
        for i in range(nbasis):
           for j in range(nbasis):
              tmp = 0.0
              for k in range(ne):
                   tmp = tmp + C[i,k]*C[j,k]
              D[i,j] = tmp
        grad_algo = nuclear_gradient(mol,center=center)
        vec = np.zeros(3)
        for i in range(0,grad_algo.shape[2]):
            vec[i%3] = vec[i%3] + np.sum(np.multiply(D,grad_algo[:,:,i]))
            print 'D',D
            print(np.multiply(D,grad_algo[:,:,i]))
        print 'elec',vec
        return vec
        

       
        
def getcentermass(mol):
       mass = np.zeros((mol.natoms,1))
       for i in range(mol.natoms):
           mass[i,0] = select_weight.get(mol.charges[0])
       return np.sum(mol.atom*mass,axis=0)/np.sum(mass)
    

def dipolemoment(mol,coef_file):
        '''This function calculates the dipole moment of a molecule'''
        #getinitertia_charge(mol.atom,mol.charges,mol.natoms)
        center = getcentermass(mol)
        center=np.array([0.0,0.0,-1.64601435])
        center=np.array([0.0,0.0,0.0])
        vec = nuclear_contribution(mol,center=center)
        print vec
        center = center.reshape(1,3)
        vec = vec - 2*electronic_contribution(mol,coef_file,center=center)
        print 'au',vec
        au_to_debye = 2.541746230211
        print vec*au_to_debye
        print vec/0.393456
        return vec


def nuclear_contribution(mol,center=np.array([0.0,0.0,0.0])):
        vec = np.zeros(3)
        for i in range(mol.natoms):
             vec = vec + mol.charges[i]*(mol.atom[i]-center)
             #vec = vec + mol.charges[i]*mol.atom[i]
        print 'nuc',vec
        return vec
