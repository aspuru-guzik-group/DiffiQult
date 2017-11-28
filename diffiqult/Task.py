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

class Tasks(object):
     '''This class manage the implemented tasks over a system included
        in DiffiQult.

        Attributes:
        
         sys  : System_mol object
               Contains the basis functions, geometry and information of the a molecular system.
         name : string
               An id of the task, it is used as prefix for outputs.
         verbose: bool
               It defines the output and screen options.

               True it prints in screen all details and a file "name.out".
         status: bool
               Keep track of the success of the different task.

               True the SCF and/or the optimization converged.

     '''
     def __init__(self,mol,name,verbose=False):
          '''
          Initialize task object that contains the inital parameters.

          Parameters:

             mol  : System_mol object
                    Contains the basis functions, geometry and information of the a molecular system.
             name : string
                    An id of the task, it is used as prefix for outputs.

          Options:
             verbose : bool
                     It defines the output and screen options.

                     True it prints in screen all details and a file "name.out"
             status: bool
                     Keep track of the success of the different task.

                     True the SCF and/or the optimization converged.
          '''
          self.name = name
          self.sys = mol
	  self.verbose = verbose
          self.status = True
	  if verbose:
	  	self.tape = open(name+'.out',"w")
		self._printheader()
          self._select_task ={
              "Energy": self.energy,
              "Opt": self.optimization,
              "Grad": self._energy_gradss,
          }
          self.select_method ={
              'BFGS': self._BFGS,
          }
          self.ntask = 0
          return

     def _printheader(self):
	 """This function prints the header of the outputfile"""
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


     def _energy_gradss(self,argnum,max_scf=301,max_d=300,printguess=None,name='Output.molden',output=False,order='first'):
       """This function returns the gradient of args"""
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
  
     def _singlepoint(self,max_scf=300,max_d=300,printcoef=False,name='Output.molden',output=False):
          """
          This function calculates a single point energy
          max_scf -> Maximum number of SCF cycles
          max_d ->  Maximum cycles of iterations if cannonical purification
          """
          log = True # We are not using logarithms of alphas
          eigen = True # We are using diagonalizations
          readguess = False #By now, we are stating the densisty matrix from scratch

       	  rguess = None
          if printcoef:
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
          if self.verbose:
             self.tape.write(' ---End--- \n')
             self.tape.write(' Time %3.7f :\n'%timer)

          if (ene == 99999):
             if self.verbose:
                 self.tape.write(' SCF did not converged :( !!\n')
             print(' SCF did not converged :( !! %s\n')
             self.status = False
          else:
             if self.verbose:
                self.tape.write(' SCF converged!!\n')
                self.tape.write(' Energy: %3.7f \n'%ene)
                if pguess != None:
                   self.tape.write(' Coefficients in file: %s\n'%pguess)
             print(' SCF converged!!')
             print(' Energy: %3.7f'%ene)
          if output:
             if self.verbose:
                self.tape.write(' Result in file: %s\n'%name)
             else:
                print(' Result in file: %s\n'%name)
          return ene 


     def _BFGS(self,ene_function,grad_fun,args,argnums,log,name,**kwargs):
          """ This function use the BFGS method implemented intialially in scipy to perform the optimization """
          print('Minimizing BFGS ...')
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
         """This function returns a list with functions that extracts the gradient of the values
         defined by args, args has the position of the inputs of energy
         """
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
                 raise NotImplementedError("Optimization is just recticted to contraction coefficients, exponents and Gaussian centers ")
          return
          
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
          name=taskname
          record = False
          maxsteps = 100
          rhf_old = 1000
          grad_fun = []
          lbda = 1.0
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
         # Pending unit tests
         if coef_file == None:
             coef_file = name
             ene = self._singlepoint(max_scf,max_scf,coef_file,name,False)
         dipolemoment(self.sys,coef_file+'.npy')
	 return

     def optimization(self,max_scf=100,log=True,scf=True,name='Output',readguess=None,output=False,argnum=[0],**kwargs):
        '''
        This function handdles the optimization procedure.

        Options:

        argnum : list of integers
                Parameter to optimize

                0:widths

                1: contraction coefficients

                2:Gaussian centers

                e.g. [0,1] to optimized widhts and contraction coefficients
        max_scf : integer
                 Maximum number of scf steps, default 30
        log     : bool
                 If we are not optimizing the log of exponent, we highly recoment leave it as True, the default.
        name    : str
                 Output file name default Output
        readguess : str
                 File path to a npy file in case on predefined initial guess of the density matrix  
        output : bool
                 True if it will print a molden file in case of success
                 
        '''
        name = self.name+'-task-'+str(self.ntask)
        if self.verbose:
           self.tape.write(' Outputfiles prefix: %s'%name)
        res,timer = self._optimization(max_scf,log,scf,readguess,argnum,taskname=name,**kwargs)
        if self.verbose:
               self._optprintres(res,timer)
        return


     def energy(self,max_scf=30,max_d=300,printguess=None,output=False,**kwargs):
        '''
        This function handdles the single point calculations

        Options:

        max_scf : integer
                 Maximum number of scf steps, default 30.
        printguess : str 

                 File path if it is requiered to prepare an inital guess for the molecular orbital coefficients.

        output : bool

                 True if it will print a molden file in case of success.
        '''
        name = self.name+'-'+str(self.ntask)
        if self.verbose:
           self.tape.write(' Output molden file: %s'%name)
        self._singlepoint(max_scf,max_d,printguess,name,output)
        return

     def runtask(self,task,**kwargs):
        ''' 
        This method run a given task and if it has success, it uptates system with the most recent energy value and basis function

        Parameters:

          task : str
               If defines the task:

              'Energy' is a single point calculation.

              'Opt' an optimization of a given parameter.

        Options:
              Check documentation for each task

        Returns:

          success : bool
                 True if task ended successfully.
        '''
        print(' Task: %s'%task)
        self.ntask += 1
        if self.verbose:
           self.tape.write(' -------------------------------------------------------- \n')
           self.tape.write(' Task: %s \n'%task)
    	function = self._select_task.get(task,lambda: self.tape.write(' This task is not implemented\n'))
        if self.verbose:
            self.tape.write('\n')
	function(**kwargs)
        return self.status

     def end(self):
        if self.verbose:
             self._printtail()
             self.tape.close()
	return
	 

def main():
     from Basis import basis_set_3G_STO as basis
     d = -1.64601435
     mol = [(1,(0.0,0.0,0.20165898)),(1,(0.0,0.0,d))]
     ne = 2
    
     system = System_mol(mol,                                ## Geometry
                         basis,                              ## Basis set (if shifted it should have the coordinates too)
                         ne,                                 ## Number of electrons
                         shifted=False,                      ## If the basis is going to be on the atoms coordinates 
                         angs=False,                         ## Units -> Bohr
                         mol_name='agua')                    ## Units -> Bohr
 
     manager = Tasks(system,
                     name='../testfiles/h2_sto_3g',      ## Prefix for all optput files
                     verbose=True)          ## If there is going to be an output

    
     manager.runtask('Energy',
                     max_scf=50,
                     printcoef=True,
                     name='../testfiles/Output.molden',
                     output=True)

     manager.runtask('Opt',
                     max_scf=50,
                     printcoef=False,
                     argnum=[2],
                     output=True)

     manager.runtask('Opt',
                     max_scf=50,
                     printcoef=False,
                     argnum=[0],
                     output=True)

     manager.runtask('Opt',
                     max_scf=50,
                     printcoef=False,
                     argnum=[0],
                     output=True)

     manager.end()
     return


if __name__ == "__main__":
     main()

