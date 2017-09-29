import Energy
from Energy import rhfenergy, penalty_inverse
from scipy.optimize import optimize as opt
import algopy
from algopy import UTPM, zeros
import numpy as np
import Minimize
from Minimize import minimize
'''

This module contain manages all tasks:
-Single point calculations.
-Optimizations.
-Gradients.

'''
class getbasis:
    '''
    This class loads the basis
    '''
    def __init__(self,molecule,basis,shifted=False):
       # It is just a class for know, just in case we want to do more pre-procesing
       self.alpha = [] # exponents
       self.coef = []  # coef
       self.xyz = []   # coordinates of atoms
       self.l = []     # distribution of angular momentums
       self.tot_prim = 0
       if shifted:
          self.get_shiftedbasis(basis)
       else:
          self.get_centeredbasis(molecule,basis)
    
    def get_centeredbasis(self,molecule,basis):
       for atom in molecule:
          for contr in basis[atom[0]]:
              for prim in contr[1]:
                  for l in self.getl_xyz(contr[0]): 
                      self.alpha.append(prim[0])
                      self.coef.append(prim[1])
                      self.l.append(l)
                      self.tot_prim += len(self.l[len(self.l)-1])
                      self.xyz.append([atom[1][0],atom[1][1],atom[1][2]])

    def get_shiftedbasis(self,basis):
       for gauss in basis:
           self.alpha.append(gauss[1])
           self.coef.append(gauss[2])
           self.l.append(gauss[0])
           self.xyz.append(gauss[3])
       return

    def getl_xyz(self,l):
       angular = {'S':[(0,0,0)],
                  'P':[(1,0,0),(0,1,0),(0,0,1)]}
       return angular[l]

class getgeom:
    '''
    This class loads the moleculegeom
    '''
    def __init__(self,molecule):
       self.xyz = []   # coordinates of atoms
       self.charge = []   # coordinates of atoms
       for atom in molecule:
          self.charge.append(int(atom[0]))
          self.xyz.append([atom[1][0],atom[1][1],atom[1][2]])
       return

    
def autochemss(task,mol,basis_set,ne,max_scf=300,max_d=300,verbose=False,tol=1e-10,scf=False,name='Output',scfout=False,argnum=[0],
               printguess=False,readguess=False,shifted=False,method='BFGS',alpha=None,penalize=None,log=True):
    '''
    This function manage several tasks
    '''
 
    ## Load basis set and xyz
    Basis = getbasis(mol,basis_set,shifted=shifted)      # Get basis
    alpha = np.array(Basis.alpha)        # alpha
    xyz = np.array(Basis.xyz)            # xyz of gaussians
    coef = np.array(Basis.coef)          # coef of primitives
    l = Basis.l                          # angular momentum
    nbasis = len(alpha)

    ## Load xyz and charges of atoms
    Geom = getgeom(mol)                              # Get geometry
    xyzatom = np.array(Geom.xyz)                     # Atom positions
    chargeatom = np.array(Geom.charge)               # Charges of atoms
    natoms = len(chargeatom)


    if log:
        alpha = np.log(Basis.alpha)

 
    def objective():
       '''This function returns the objective function to minimize, it can be the plain energy 
       or we can add a penalization, at the moment only the first case has been extensively tested '''
       record = False
       if penalize == None:
          ene_function = rhfenergy
       else:
           ene_function = select_penalize.get(penalize,lambda: penalty_inverse)
           args = [alpha,coef,xyz,l,chargeatom,xyzatom,natoms,nbasis,ne,max_scf,max_d,lbda,log,scf,printguess,readguess,name,record,alpha]
       return ene_function(*(args))
        
    def energyss(output=scfout,printguess=printguess):
       '''This function calculates a single point energy'''
       rguess = None
       if printguess:
          pguess = name+'data_guess_C.npy'
       else:
          pguess = None
       args=[alpha,coef,xyz,l,chargeatom,xyzatom,natoms,nbasis,ne,max_scf,max_d,log,scf,pguess,rguess,name,output,alpha]
       print 'Single point ...'
       ene = rhfenergy(*(args))
       print 'SCF converged!!'
       print 'Energy: ',ene
       return ene

    def energy_gradss():
       '''This function returns the gradient of args'''
       ## For the moment it retuns a value at a time
       ## This is used only by testing functions.
       grad_fun =[]
       for i in argnum:
           var = UTPM.init_jacobian(args[i])
           diff_args = list(args)              # We are making a copy of args
           diff_args[i] = var
           diff_args[-1]= var
           return UTPM.extract_jacobian(rhfenergy(*(diff_args)))

    def algo_gradfun(function,args):
       '''This function returns a list with functions that extracts the gradient of the values
       defined by args, args has the position of the inputs of energy '''
       grad_fun =[]
       def function_builder(narg):
           def algo_jaco(*args, **kwargs):
               var = UTPM.init_jacobian(args[narg])
               diff_args = list(args)              # We are making a copy of args
               diff_args[narg] = var
               diff_args[-1]= var
               diff_args = tuple(diff_args)
               return UTPM.extract_jacobian(rhfenergy(*(diff_args)))
           return algo_jaco
       for i in argnum:
           grad_fun.append(function_builder(i))
       return grad_fun
     
    def optimizationss():
       record = False
       maxsteps = 100
       rhf_old = 1000
       grad_fun = []
       lbda = 1.0
       ##`alpha = np.array(Basis.alpha)        # alpha
       ## optimizing from a guess.
       rguess = None
       if readguess:
           pguess = name+'data_guess_C.npy'
           energyss(output=False,printguess=True)
           rguess= pguess
       ## Selecting penalized methods
       if penalize == None:
          record = False
          ene_function = rhfenergy
          pguess = None
          args=[alpha,coef,xyz,l,chargeatom,xyzatom,natoms,nbasis,ne,max_scf,max_d,log,scf,pguess,rguess,name,record,alpha]
       else:
          print 'Check this'
          exit()
          record = False
          ene_function = select_penalize.get(penalize,lambda: penalty_inverse)
          args = [alpha,coef,xyz,l,chargeatom,xyzatom,natoms,nbasis,ne,max_scf,max_d,lbda,log,scf,name,record]
       grad_fun = algo_gradfun(ene_function,args)  ## This defines de gradient function of autograd
       opt = select_method.get(method,lambda: BFGSss)  ## This selects the opt method
       res = opt(ene_function,grad_fun,args,argnum)       
       return res
       

    def BFGS(ene_function,grad_fun,args,argnums):
       print 'Minimizing BFGS ...'
       G = False
       var = [args[i] for i in argnums] ## Arguments to optimize
       for i in reversed(argnums):
           del(args[i])                 ## Getting the rest of them
       if log:
          tol = 1e-05 
       else:
          tol = 1e-07 
       terms =  minimize(ene_function,var,
                         args=tuple(args),
                         argnum=argnums,
                         method='BFGS',jac=grad_fun,gtol=tol,name=name,options={'disp': True})
       return 0.0#terms
       
    def Newton(ene_function,grad_fun,argsl,argnums):
       tool = 1e-6
       max_iter = 30
       if len(argnums) == 0:
          print 'I don not opt more than one variable'
          exit()
       for i in argnums:
           hess_fun = hessian(ene_function,argnum=i)

       step = 0
       name_tmp = name+'_newton_step_'+str(step)
       record_tmp = False
       args=[alpha,coef,xyz[0],xyz[1],xyz[2],l,chargeatom,xyzatom[0],xyzatom[1],xyzatom[2],natoms,nbasis,ne,max_scf,max_d,log,record_tmp,name_tmp,record_tmp]
       x_old = alpha
       ene_old = rhfenergy(*(args))
       for i in range(max_iter):
          print 'STEP: ',step,ene_old
          record_tmp = False
          argsl[argnums[0]] = x_old
          for i in grad_fun:
              g = i(*(argsl))
          hess = np.linalg.inv(hess_fun(*(argsl)))
          x_new = x_old - np.dot(hess,g)
          name_tmp = name+'_newton_step_'+str(step)
          record_tmp = True
          args=[alpha,coef,xyz[0],xyz[1],xyz[2],l,chargeatom,xyzatom[0],xyzatom[1],xyzatom[2],natoms,nbasis,ne,max_scf,max_d,log,record_tmp,name_tmp,record_tmp]
          argsl[argnums[0]] = x_new
          args[argnums[0]] = x_new
          ene_new = rhfenergy(*(args))
          err = abs(ene_new-ene_old)
          step += 1
          if err < tool:
              return 0.0
          ene_old = ene_new 
          x_new = x_old

       print 'This did not converge' 
       exit()
       return 0.0


    select_penalize={
          "Inverse": penalty_inverse,
          }
    select_task ={
          "Energy": energyss,
          'Opt': optimizationss,
          'Grad': energy_gradss,
          'Objective': objective,
          }

    select_method ={
          'BFGS': BFGS,
          'Newton': Newton,
          }

    energy = select_task.get(task,lambda: "nothing")
    ene = energy()
    print 'TASK DONE'
    print '___________________________________'
    return ene
