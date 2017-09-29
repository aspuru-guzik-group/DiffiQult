from Energy import rhfenergy, penalty_inverse
from scipy.optimize import optimize as opt
from Dipole import dipolemoment
from Minimize import minimize
from Molecule import Getbasis,Getgeom,System_mol

import numpy as np
import time
import algopy
from algopy import UTPM, zeros
'''

This module contain manages all tasks:
-Single point calculations.
-Optimizations.
-Gradients.

'''


class _SelectTaskError(RuntimeError):
    pass

class Gradients(object):

     def __init__(self,function,args,argnum):
         self.function = function
         self.args = args
         self.argnum = argnum
         return
         

     def _algo_gradfun(self):
         '''This function returns a list with functions that extracts the gradient of the values
         defined by args, args has the position of the inputs of energy '''
         grad_fun =[]
         def function_builder(narg):
            def algo_jaco(*args, **kwargs):
                var = UTPM.init_jacobian(self.args[self.narg])
                diff_args = list(args)              # We are making a copy of args
                diff_args[narg] = var
                diff_args[-1]= var
                diff_args = tuple(diff_args)
                return UTPM.extract_jacobian(rhfenergy(*(diff_args)))
            return algo_jaco
         for i in argnum:
           grad_fun.append(function_builder(i))
	 return grad_fun


     def _algo_hessfun(self,function,args,argnum):
         '''This function returns a list with functions that extracts the hessian of the values
         defined by args, args has the position of the inputs of energy '''
         grad_fun =[]
         def function_builder(narg):
            def algo_jaco(*args, **kwargs):
                var = UTPM.init_hessian(args[narg])
                diff_args = list(args)              # We are making a copy of args
                diff_args[narg] = var
                diff_args[-1]= var
                diff_args = tuple(diff_args)
                return UTPM.extract_hessian(rhfenergy(*(diff_args)))
            return algo_jaco
         for i in argnum:
           grad_fun.append(function_builder(i))
	 return grad_fun


class Autochem(object):
     ''' This class manage several tasks'''
     def __init__(self,mol,basis_set,ne,name,verbose=False,shifted=False):
	  self.sys = System_mol(mol,basis_set,ne,'Molecule',shifted=shifted)
          self.name = name
	  self.verbose = verbose
          self.status = True
	  if verbose:
	  	self.tape = open(name+'.out',"w")
		self._printheader()
          self._select_task ={
              "Energy": self.energy,
              "Opt": self.optimization,
              "Grad": self.energy_gradss,
              #"Objective": objective,
              "Dipole": self.dipole
          }
          self.select_method ={
              'BFGS': self._BFGS,
              'Newton': Newton,
          }
          return

     def _printheader(self):
	 '''This function prints the header of the outputfile'''
         self.tape.write(' *************************************************\n')
         self.tape.write(' DiffiQult Ago 2017\n')
         self.tape.write(' Author: Teresa Tamayo Mendoza \n')
         self.tape.write(' *************************************************\n\n')
         localtime = time.asctime( time.localtime(time.time()) )
	 self.tape.write(" Starting at %s\n"%localtime)
        

    
     def _printtail(self):
         localtime = time.asctime( time.localtime(time.time()) )
	 self.tape.write(" Finishing at %s \n"%localtime)
         self.tape.write(' *************************************************\n\n')
	 return


     def energy_gradss(self,argnum,max_scf=301,max_d=300,printguess=None,name='Output.molden',output=False,order='first'):
       '''This function returns the gradient of args'''
       ## For the moment it retuns a value at a time
       ## This is used only by testing functions.
       eigen = True
       rguess = False
       args=[np.log(self.sys.alpha),self.sys.coef,self.sys.xyz,self.sys.l,self.sys.charges,self.sys.atom,self.sys.natoms,self.sys.nbasis,
               self.sys.list_contr,self.sys.ne,
               max_scf,max_d,log,eigen,None,None,
               name,output,self.sys.alpha] # Last term is only used for Algopy
       if self.verbose:
             self.tape.write(' \n Grad point ...\n')
             self.tape.write(' ---Start--- \n')
             self.tape.write(' Initial parameters \n')
             self.tape.write(' Maximum number of SCF: %d\n'%max_scf)
             self.tape.write(' Default SCF tolerance: %f\n'%1e-8)
             self.tape.write(' Initial density matrix: %s\n'%str(rguess))
             self.sys.printcurrentgeombasis(self.tape)
       grad_fun =[]
       for i in argnum:
           var = UTPM.init_jacobian(args[i])
           diff_args = list(args)              # We are making a copy of args
           diff_args[i] = var
           diff_args[-1]= var
           t0 = time.clock()
           grad = UTPM.extract_jacobian(rhfenergy(*(diff_args)))
           timer = time.clock() - t0
           self.sys.grad = grad
           self.tape.write(' ---End--- \n')
           self.tape.write(' Time %3.7f :\n'%timer)
       return grad
  
     def _singlepoint(self,max_scf=300,max_d=300,printguess=None,name='Output.molden',output=False):
          '''This function calculates a single point energy
          max_scf -> Maximum number of SCF cycles
          max_d ->  Maximum cycles of iterations if cannonical purification
          '''
          log = True # We are not using logarithms of alphas
          eigen = True # We are using diagonalizations
          readguess = False #By now, we are stating the densisty matrix from scratch

       	  rguess = None
          if printguess:
             pguess = name+'.npy'
          else:
             pguess = None

       	  args=[np.log(self.sys.alpha),self.sys.coef,self.sys.xyz,self.sys.l,self.sys.charges,self.sys.atom,self.sys.natoms,self.sys.nbasis,
               self.sys.list_contr,self.sys.ne,
               max_scf,max_d,log,eigen,pguess,rguess,
               name,output,self.sys.alpha] # Last term is only used for Algopy

	  if self.verbose:
             self.tape.write(' \n Single point ...\n')
             self.tape.write(' ---Start--- \n')
             self.tape.write(' Initial parameters \n')
             self.tape.write(' Maximum number of SCF: %d\n'%max_scf)
             self.tape.write(' Default SCF tolerance: %f\n'%1e-8)
             self.tape.write(' Initial density matrix: %s\n'%str(rguess))
             self.sys.printcurrentgeombasis(self.tape)

          # Function         
          t0 = time.clock()
          ene = rhfenergy(*(args))
          timer = time.clock() - t0

          self.energy = ene
          self.tape.write(' ---End--- \n')
          self.tape.write(' Time %3.7f :\n'%timer)
          if (ene == 99999):
             self.tape.write(' SCF did not converged :( !! %s\n'%output)
             self.status = False
          else:
             self.tape.write(' SCF converged!!\n')
             self.tape.write(' Energy: %3.7f \n'%ene)
             if output:
                self.tape.write(' Result in file: %s\n'%name)
             if pguess != None:
                self.tape.write(' Coefficients in file: %s\n'%pguess)
          return ene 


     def _BFGS(self,ene_function,grad_fun,args,argnums,log,name,**kwargs):
          print 'Minimizing BFGS ...'
          G = False
          var = [args[i] for i in argnums] ## Arguments to optimize
          for i in reversed(argnums):
               del(args[i])                 ## Getting the rest of them
          if log and 0 in argnums:
             tol = 1e-05   #It basically depends if we are using log or not for alpha
          else:
             tol = 1e-07
          terms =  minimize(ene_function,var,
                         args=tuple(args),
                         argnum=argnums,
                         method='BFGS',jac=grad_fun,gtol=tol,name=name,options={'disp': True},**kwargs)
          return terms 

     def _algo_gradfun(self,function,args,argnum):
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


     def _algo_hessfun(self,function,args,argnum):
         '''This function returns a list with functions that extracts the hessian of the values
         defined by args, args has the position of the inputs of energy '''
         grad_fun =[]
         def function_builder(narg):
            def algo_jaco(*args, **kwargs):
                var = UTPM.init_hessian(args[narg])
                diff_args = list(args)              # We are making a copy of args
                diff_args[narg] = var
                diff_args[-1]= var
                diff_args = tuple(diff_args)
                return UTPM.extract_hessian(rhfenergy(*(diff_args)))
            return algo_jaco
         for i in argnum:
           grad_fun.append(function_builder(i))
	 return grad_fun


     def _optupdateparam(self,argnum,x):
          ### HARD CODED (ONLY WORKS WITH ALPHA AND XYZ)
          cont = 0
          for i in argnum:
             if i == 0:
                 self.sys.alpha = np.exp(x[cont:cont+len(self.sys.alpha)])
                 cont += len(self.sys.alpha)
             elif i == 2:
                 self.sys.xyz = x[cont:cont+self.sys.nbasis*3].reshape(self.sys.nbasis,3) 
                 cont += self.sys.nbasis*3
             elif i == 1:
                 self.sys.coef = x[cont:cont+len(self.sys.alpha)] 
                 cont += self.sys.alpha
             else:
                 print('FIX ME :(')
                 exit()
          
     def _optprintres(self,res,timer):
          self.tape.write(' ---End--- \n')
          self.tape.write(' Time %3.7f :\n'%timer)
          self.tape.write(' Message: %s\n'%res.message)
          self.tape.write(' Current energy: %f\n'%res.fun)
          self.tape.write(' Current gradient % f \n'%np.linalg.norm(res.jac,ord=np.inf))
          self.tape.write(' Number of iterations %d \n'%res.nit)
          self.sys.printcurrentgeombasis(self.tape)

     def _optprintparam(self,max_scf,rguess):
          self.tape.write(' Initial parameters \n')
          self.tape.write(' Maximum number of SCF: %d\n'%max_scf)
          self.tape.write(' Default SCF tolerance: %f\n'%1e-8)
          self.tape.write(' Initial density matrix: %s\n'%str(rguess))
          self.tape.write(' Maximum number of optimization steps: (FIX ME)%d\n'%30)
          self.tape.write(' Tolerance in jac infinity norm (Hardcoded): %f \n'%1e-1)
          self.tape.write(' Tolerance in energy (Hardcoded): %f \n'%1e-5)
          self.sys.printcurrentgeombasis(self.tape)

     def _optimization(self,max_scf=100,log=True,scf=True,readguess=None,argnum=[0],taskname='Output', method='BFGS',penalize=None,**otherargs):
          print 'alpha',self.sys.alpha
          name=taskname
          record = False
          maxsteps = 100
          rhf_old = 1000
          grad_fun = []
          lbda = 1.0
          ##`alpha = np.array(Basis.alpha)        # alpha
          ## optimizing from a guess.
          rguess = None
          if readguess:
             pguess = name +'.npy'
             out = False
             ene = self._singlepoint(maxsteps,maxsteps,pguess,name,False)
	     if ene == 99999:
                raise NameError('SCF did not converved')
             rguess= pguess

	  if self.verbose:
             self.tape.write(' \n Optimization ...\n')
             self.tape.write(' ---Start--- \n')
             self._optprintparam(max_scf,rguess)

          ene_function = rhfenergy
          pguess = None

          record = False
       	  args=[np.log(self.sys.alpha),self.sys.coef,self.sys.xyz,self.sys.l,self.sys.charges,self.sys.atom,self.sys.natoms,self.sys.nbasis,
               self.sys.list_contr,self.sys.ne,
               max_scf,max_scf,log,True,pguess,rguess,
               name,record,self.sys.alpha] # Last term is only used for Algopy
          grad_fun = self._algo_gradfun(ene_function,args,argnum)  ## This defines de gradient function of autograd
          opt = self.select_method.get(method,lambda: self._BFGS)  ## This selects the opt method
          t0 = time.clock()
          res = opt(ene_function,grad_fun,args,argnum,log,name,**otherargs)
          timer = time.clock()-t0
          self._optupdateparam(argnum,res.x)
          self.status = bool(res.status == 0 or res.status==1)
          return res,timer

     def dipole(self,coef_file=None,max_scf=100,name='Output',**kwargs):
         if coef_file == None:
             coef_file = name
             ene = self._singlepoint(max_scf,max_scf,coef_file,name,False)
         dipolemoment(self.sys,coef_file+'.npy')
	 return

     def optimization(self,ntask=0,max_scf=100,log=True,scf=True,name='Output',readguess=None,output=False,argnum=[0],**kwargs):
        '''This function handdles the single point calculations '''
        name = self.name+'-task-'+str(ntask)
        if self.verbose:
           self.tape.write(' Outputfiles prefix: %s'%name)
        res,timer = self._optimization(max_scf,log,scf,readguess,argnum[ntask],taskname=name,**kwargs)
        if self.verbose:
               self._optprintres(res,timer)
          
	#except:
	#    self.tape.write('An error occured in the Optimization \n')
        return


     def energy(self,ntask=0,max_scf=300,max_d=300,printguess=None,output=False,**kwargs):
        '''This function handdles the single point calculations '''
        name = self.name+'-'+str(ntask)
        if self.verbose:
           self.tape.write(' Output molden file: %s'%name)
        self._singlepoint(max_scf,max_d,printguess,name,output)
        return

     def runtask(self,task,ntask=0,**kwargs):
        self.tape.write(' -------------------------------------------------------- \n')
        self.tape.write(' Task: %s \n'%task)
    	function = self._select_task.get(task,lambda: self.tape.write(' This task is not implemented\n'))
        self.tape.write('\n')
	function(ntask,**kwargs)

     def end(self):
        if self.verbose:
             self._printtail()
             self.tape.close()
	return
	 

def autochem(tasks,mol,basis_set,ne,name='Output',verbose=True,shifted=False,**kwargs):
#max_scf=300,max_d=300,verbose=False,tol=1e-10,scf=False,name='Output',scfout=False,argnum=[0],
#               printguess=False,readguess=False,shifted=False,method='BFGS',alpha=None,penalize=None,log=True):

     adchem = Autochem(mol,basis_set,ne,name,verbose,shifted=shifted)
     for i,task in enumerate(tasks):
         adchem.runtask(task, ntask = i,**kwargs)
         if not adchem.status:
            break
     adchem.end()
     return adchem
         
