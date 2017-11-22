import Integrals
from Integrals import erivector,overlapmatrix,nuclearmatrix,kineticmatrix,normalization
import Tools
from Tools import euclidean_norm,printmatrix,eri_index
import algopy
from algopy import UTPM, zeros,transpose
import numpy as np
import Eigen 
from Eigen import eigensolver

'''
This module contains functions to obtain energy
'''


def mo_naturalorbital(D):
    eigsys = eigensolver(D)
    print eigsys[0]
    print eigsys[1]
    return [eigsys[0]] 
    

def nuclearrepulsion(atoms,charges,natoms):
    tmp = 0.0
    for i, xyza in enumerate(atoms):
       for j in range(i+1,natoms):
           xyzb = atoms[j]
           tmp = tmp + charges[i]*charges[j]/euclidean_norm(np.subtract(xyzb,xyza))
    return tmp

def fockmatrix(Hcore,Eri,D,nbasis,alpha,dtype):
    F = algopy.zeros((nbasis,nbasis),dtype=dtype)
    for i in range(nbasis):
       for j in range(i,nbasis):
           tmp = Hcore[i,j]
           for k in range(nbasis):
               for l in range(nbasis):
                 tmp = D[k,l]*(2.0*Eri[eri_index(i,j,k,l,nbasis)]-Eri[eri_index(i,k,j,l,nbasis)])+ tmp
           F[i,j] = F[j,i] = tmp
    return F


def cannonicalputication(D,S):
    D2 = np.dot(np.dot(D,S),D)
    D3 = np.dot(np.dot(D2,S),D)
    c = np.trace(np.subtract(D2,D3))/np.trace(np.subtract(D,D2)) ## Eq 17 Manolopoulos
    
    if c <= 0.5:
        tmp =  np.subtract(np.add(np.multiply((1.0-2.0*c),D),np.multiply((1.0+c),D2)),D3)/(1.0-c)
    else:
        tmp =  np.subtract(np.multiply((1.0+c),D2),D3)/(c)
    
    return np.divide(tmp,np.trace(np.dot(D,S))) ## I am not dividing by one here!!
    
def newdensity(F,Sinv,nbasis,ne):
    F_offdiag = []
    for i in range(nbasis):
        F_offdiag.append(0.0) 
    for i in range(nbasis):
       for j in range(nbasis):
           if i != j:
              F_offdiag[i] = np.absolute(F[i,j]) + F_offdiag[i]
    listFmin = []
    listFmax = []
    for i in range(nbasis):
        listFmin.append(F[i][i] - F_offdiag[i])
        listFmax.append(F[i][i] + F_offdiag[i])
    
    Fmax = max(listFmax)
    Fmin = min(listFmin)

    mubar = np.trace(F)/nbasis
    lamb = min([(nbasis-ne)/(mubar-Fmin),ne/(Fmax-mubar)])

    return np.add(np.multiply(lamb/nbasis,np.subtract(mubar*Sinv,np.dot(np.dot(Sinv,F),Sinv))),np.multiply(np.float64(1.0*ne/nbasis),Sinv))

def rhfenergy(alpha_old,coef2,xyz,l,charges,xyz_atom,natoms,nbasis,contr_list,ne,max_scf,max_d,log,eigen,printguess,readguess,name,write,dtype):
    '''
    Here will be the rhf function
    eigen = False -> Canonical Purification
    tape = name -> Tape for recording
    record = bool -> Are we recording?
    readguess -> are we reading a guess?, name of the file
    printguess -> are we creating a guess?, name of the file
    name -> Name of record (it must be -3)
    write -> Record output.molden (it must be -2)
    dtype -> type of var to differentiate for Algopy empty matrices
    '''
    tool_D = 1e-8
    tool = 1e-8
    
    if log:
       alpha = algopy.exp(alpha_old)
    else:
    	alpha = alpha_old

    if type(alpha) != np.ndarray: ## Cover the case of diff xyz atom
        coef = normalization(alpha,coef2,l,contr_list,dtype=dtype)
    else:
        coef = normalization(alpha,coef2,l,contr_list,dtype=np.float64(1.0))
    
    V = nuclearmatrix(alpha,coef,xyz,l,nbasis,charges,xyz_atom,natoms,contr_list,dtype=dtype)
    if type(xyz_atom) != np.ndarray: ## Cover the case of diff xyz atom
        dtypef = np.float64(1.0)
        S = overlapmatrix(alpha,coef,xyz,l,nbasis,contr_list,dtype=dtypef)
        T = kineticmatrix(alpha,coef,xyz,l,nbasis,contr_list,dtype=dtypef)
        Eri = erivector(alpha,coef,xyz,l,nbasis,contr_list,dtype=dtypef)
   
    else:
        S = overlapmatrix(alpha,coef,xyz,l,nbasis,contr_list,dtype=dtype)
        T = kineticmatrix(alpha,coef,xyz,l,nbasis,contr_list,dtype=dtype)
        Eri = erivector(alpha,coef,xyz,l,nbasis,contr_list,dtype=dtype)
    Hcore = T + V
    if eigen:
       eigsys = eigensolver(S)
       SqrtLambda = algopy.diag(1./algopy.sqrt(eigsys[0]))
       L = eigsys[1]
       LT = algopy.transpose(L)
       SqrtS = algopy.dot(algopy.dot(L, SqrtLambda), LT)
       SqrtST = algopy.transpose(SqrtS)
    else:
       Sinv = np.linalg.inv(S)
 
    if readguess != None:
        #print 'Reading previous guess'+readguess
	C = np.load(readguess)
        D = np.zeros((nbasis,nbasis))
        for i in range(nbasis):
           for j in range(nbasis):
              tmp = 0.0
              for k in range(ne):
                   tmp = tmp + C[i,k]*C[j,k]
              D[i,j] = tmp
        F = fockmatrix(Hcore,Eri,D,nbasis,alpha,dtype)
    else:
    	F = Hcore
    OldE = 1e8

    status = False
    E_step = []
    for scf_iter in range(max_scf):
        if eigen:
           Fprime = algopy.dot(algopy.dot(SqrtST,F), SqrtS)
           eigsysFockOp = eigensolver(Fprime)
           Cprime = eigsysFockOp[1]
           C = algopy.dot(SqrtS, Cprime)
           Fprime = algopy.dot(algopy.dot(SqrtST,F), SqrtS)
           eigsysFockOp = eigensolver(Fprime)
           Cprime = eigsysFockOp[1]
           C = algopy.dot(SqrtS, Cprime)
           D = algopy.zeros((nbasis,nbasis),dtype=dtype)
           for i in range(nbasis):
              for j in range(nbasis):
                 tmp = 0.0
                 for k in range(ne):
                      tmp = tmp + C[i,k]*C[j,k]
                 D[i,j] = tmp
        else:
           D = newdensity(F,Sinv,nbasis,ne)
           for i in range(max_d):
              D = cannonicalputication(D,S)
              err = np.linalg.norm(D - np.dot(np.dot(D,S),D))
              if err < tool_D:
                break
        F = fockmatrix(Hcore,Eri,D,nbasis,alpha,dtype)
        E_elec = algopy.sum(np.multiply(D,Hcore+F))
        E_step.append(E_elec)
        E_nuc = nuclearrepulsion(xyz_atom,charges,natoms)
        if np.absolute(E_elec - OldE) < tool:
           status = True 
           break
        OldE = E_elec
    E_nuc = nuclearrepulsion(xyz_atom,charges,natoms)
    if printguess !=None:
        np.save(printguess, C)

    def update_system():
       mol.energy = E_elec + E_nuc
       mol.erepulsion = Eri
       mol.hcore = Hcore
       mol.mo_coeff = C
       return 
            

    def write_molden():
       import Data
       from Data import select_atom
       ## Details of calculation
       tape.write('[Energy] \n')
       tape.write('E_elec: '+str(E_elec)+'\n')
       tape.write('E_nuc: '+str(E_nuc)+'\n')
       tape.write('E_tot: '+str(E_nuc+E_elec)+'\n')
       tape.write('SCF Details\n')
       line = 'Eigen: '
       if eigen: 
           tape.write(line+'True')
       else:
           tape.write(line+'False')
       tape.write('\n')
       for i, step in enumerate(E_step):
           line = 'Step: '+str(i)+' '+str(step)
           tape.write(line+'\n')

       ### C Matrix
       tape.write('[CM] \n')
       tape.write('C AO times MO\n')
       printmatrix(C,tape)

       ### D Matrix
       tape.write('[DM] \n')
       tape.write('D \n')
       printmatrix(D,tape)

       ### D Matrix
       tape.write('[NMO] \n')
       tape.write('NMO \n')
       printmatrix(mo_naturalorbital(D),tape)

       ### MO energies
       tape.write('[MOE] \n')
       tape.write('MOE \n')
       for i,energ in enumerate(eigsysFockOp[0]):
          tape.write(str(i)+' '+str(energ)+'\n')

       ### MO energies
       tape.write('[INPUT] \n')
       line = 'mol = ['
       for i,coord in enumerate(xyz_atom):
           line += '('+str(charges[i])+','
           line +='('+ str(coord[0])+','+str(coord[1])+','+str(coord[2]) + ')),\n'
       tape.write(line)
       cont = 0
       line = 'basis = ['
       for i,ci in enumerate(contr_list):
           line +='['
           line += '('+str(l[i][0])+','+str(l[i][1])+','+str(l[i][2])+'),'
           for ii in range(ci):
               line += str(alpha[cont])+','+str(coef[i])
               line +=',('+ str(xyz[i,0])+','+ str(xyz[i,1])+','+str(xyz[i,2]) + ')],\n'
               cont +=1
           line +=']\n'
       tape.write(line)
           
       ### Atom coordinates
       tape.write('[Atoms]\n')
       for i,coord in enumerate(xyz_atom):
           line = select_atom.get(charges[i])
           line +=' '+str(i+1)+' '+str(charges[i])
           line +=' '+ str(coord[0])+' '+str(coord[1])+' '+str(coord[2]) + '\n'
           tape.write(line)
       ### Basis coordinates
       for i,coord in enumerate(xyz):
           line = 'XX'
           line +=' '+str(i+natoms+1)+' '+str(0)
           line +=' '+ str(coord[0])+' '+ str(coord[1])+' '+str(coord[2]) + '\n'
           tape.write(line)

       ### Basis set
       cont = 0
       tape.write('[GTO]\n')
       for i,ci in enumerate(contr_list):
          tape.write('  '+str(i+1+natoms)+'  0\n')
          if np.sum(l[i]) == 0 :
             tape.write(' s   '+str(ci)+' 1.0 '+ str(l[i][0])+' '+str(l[i][1])+' '+str(l[i][2])+'\n')
          else:
             tape.write(' p   '+str(ci)+' 1.0 '+ str(l[i][0])+' '+str(l[i][1])+' '+str(l[i][2])+'\n')
             #tape.write(' p   '+str(1)+' 1.0 '+ str(l[i])+'\n')
          for ii in range(ci):
              line ='  '+str(alpha[cont])+' '+str(coef[cont])+'\n'
              tape.write(line)
              cont +=1
          line ='  \n'
          tape.write(line)
       ### MOs
       tape.write('[MO]\n')
       for j in range(nbasis):
          tape.write('  Sym=  None\n')
          tape.write('  Ene=      '+str(eigsysFockOp[0][j])+'\n')
          tape.write('  Spin= Alpha\n')
          if j > ne:
             tape.write('  Occup=    0.0\n')
          else:
             tape.write('  Occup=    2.0\n')
          for i in range(nbasis):
              tape.write(str(i+1+natoms)+' '+str(C[i,j])+'\n')

    if status:
      if write:
         tape = open(name+'.molden',"w")
         write_molden()
         #update_system()
         tape.close()
    else:
       print('E_elec: '+str(E_elec)+'\n')
       print('E_nuc: '+str(E_nuc)+'\n')
       print('E_tot: '+str(E_nuc+E_elec)+'\n')
       print 'SCF DID NOT CONVERGED'
       return 99999
     
    return E_elec+E_nuc

def penalty_inverse(alpha,coef3,x,y,z,l,charges,x_atom,y_atom,z_atom,natoms,nbasis,ne,max_scf,max_d,lbda,eigen,name,write):
    energy = rhfenergy(alpha,coef3,x,y,z,l,charges,x_atom,y_atom,z_atom,natoms,nbasis,ne,max_scf,max_d,eigen,name,write)

    ## Rebuild xyz for basis
    x = np.reshape(x,(1,nbasis))
    y = np.reshape(y,(1,nbasis))
    z = np.reshape(z,(1,nbasis))
    xyz = np.concatenate((np.concatenate((x,y),axis=0),z),axis=0).T

    coef = np.ones(nbasis,dtype=dtype)
    coef = normalization(alpha,coef,l,nbasis)
    S = overlapmatrix2(alpha,coef,l,nbasis)
    #print 'This is det S',np.linalg.det(S)
    print 'This is eigen S',np.linalg.eigh(S)[0]
    penalty = lbda*1.0e-3/np.sqrt(np.linalg.det(S))
    return energy + penalty

