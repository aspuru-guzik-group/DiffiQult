import sys
sys.path.insert(0, '../')
import os
import Task
from Task import autochem, autochemss

def energy(mol,ne,basis_set,name='output',shifted=False,log=False):
    #return autochemss('Energy',mol,basis_set,ne/2,max_scf=100,max_d=300,scfout=True,scf=True,name=name,shifted=shifted,log=log)
    return autochem(['Energy'],mol,basis_set,ne/2,max_scf=100,max_d=300,scfout=True,scf=True,name=name,shifted=shifted,log=log)

def old_energy(mol,ne,basis_set,name='output',shifted=False,log=False):
    return autochemss('Energy',mol,basis_set,ne/2,max_scf=100,max_d=300,scfout=True,scf=True,name=name,shifted=shifted,log=log)
    #return autochem(['Energy'],mol,basis_set,ne/2,max_scf=100,max_d=300,scfout=True,scf=True,name=name,shifted=shifted,log=log)

def grad(mol,ne,basis_set,name='output',scf=False,argnum=[0]):
    return autochemss('Grad',mol,basis_set,ne/2,max_scf=2,max_d=300,scfout=False,scf=True,name=name,argnum=argnum)

def optimization(mol,ne,basis_set,name='output',argnum=[0],shifted=False,readguess=False):
    #return autochemss('Opt',mol,basis_set,ne/2,max_scf=100,max_d=300,scfout=True,scf=True,name=name,argnum=argnum,log=True,shifted=shifted,readguess=readguess)
    return autochem(['Opt'],mol,basis_set,ne/2,max_scf=100,max_d=300,scfout=True,scf=True,name=name,argnum=argnum,log=True,shifted=shifted,readguess=readguess)
