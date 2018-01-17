from Energy import rhfenergy, penalty_inverse
from scipy.optimize import optimize as opt
from Dipole import dipolemoment
from Minimize import minimize
from Molecule import Getbasis,Getgeom,System_mol

import sys
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

args_dict = {0: 'Exponents',
             1: 'Contraction coefficients',
             2: 'Gaussian centers',
             4: 'Geometry centers'}


def function_grads_algopy(function,argnum):
    """This function returns a list with functions that extracts the gradient of the values
    defined by args, args has the position of the inputs of function
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


def grad_evaluator_algopy(function,args,argnum, **kwargs):
    """ This function returns the gradient of a function evaluated at args
    with argumentx specify by argnum """
    list_function = function_grads_algopy(function,argnum)
    grad = []
    for function in list_function:
        grad.append(function(*args,**kwargs))
    return grad



def function_hessian_algopy(function,argnum):
    """This function returns a list with functions that extracts the hessian of the values
    defined by args, args has the position of the inputs of energy """
    grad_fun =[]
    def function_builder(narg):
       def algo_jaco(args, **kwargs):
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
              "Grad": self.gradient,
          }
          self.select_method ={
              'BFGS': self._BFGS,
          }
          self.ntask = 0
          return

     def _energy_args(self,max_scf=100,max_d=10,log=True,printguess=None,readguess=None,name='Output.molden',write=False,eigen=True):
          if log:
             alpha = np.log(self.sys.alpha)
          else:
             alpha = self.sys.alpha
          args=[alpha,self.sys.coef,self.sys.xyz,self.sys.l,self.sys.charges,self.sys.atom,self.sys.natoms,self.sys.nbasis,
               self.sys.list_contr,self.sys.ne,
               max_scf,max_d,log,eigen,
               printguess,readguess,
               name,write,np.float(1.0)] # Last term is only used for Algopy
          return args

     def _printheader(self):
	 """This function prints the header of the outputfile"""
         self.tape.write(' *************************************************\n')
         self.tape.write(' DiffiQult \n')
         self.tape.write(' Author: Teresa Tamayo Mendoza \n')
         self.tape.write(' *************************************************\n\n')
         localtime = time.asctime( time.localtime(time.time()) )
	 self.tape.write(" Starting at %s\n"%localtime)
        

     def _printtail(self):
         localtime = time.asctime( time.localtime(time.time()) )
	 self.tape.write(" Finishing at %s \n"%localtime)
         self.tape.write(' *************************************************\n\n')
	 return

     def _printenergy(self,max_scf,rguess,tol=1e-8):
         self.tape.write(' SCF Initial parameters \n')
         self.tape.write(' Maximum number of SCF: %d\n'%max_scf)
         self.tape.write(' Default SCF tolerance: %f\n'%tol)
         self.tape.write(' Initial density matrix: %s\n'%str(rguess))
         self.sys.printcurrentgeombasis(self.tape)
         return
     
     def _print_head_grad(self,max_scf,rguess):
         self.tape.write(' \n Grad single point ...\n')
         self.tape.write(' ---Start--- \n')
         self._printenergy(max_scf,rguess)

     def _print_tail_grad(self,timer,grad,argnum):
         self.tape.write(' ---End--- \n')
         for i,argn in enumerate(argnum):
              self.tape.write('  %s:\n'%args_dict[i])
              self.tape.write('  '+str(grad[i]))
         self.tape.write(' Time %3.7f :\n'%timer)

     def _print_head_energy(self,max_scf,rguess):
         self.tape.write(' \n Single point ...\n')
         self.tape.write(' ---Start--- \n')
         self._printenergy(max_scf,rguess)

     def _print_tail_energy(self,timer,ene,pguess,output=False,name='Output.molden'):
          self.tape.write(' ---End--- \n')
          self.tape.write(' Time %3.7f :\n'%timer)
          if (ene == 99999):
             self.tape.write(' SCF did not converged :( !!\n')
          else:
             self.tape.write(' SCF converged!!\n')
             self.tape.write(' Energy: %3.7f \n'%ene)
             if pguess != None:
                self.tape.write(' Coefficients in file: %s\n'%pguess)
          if output:
             self.tape.write(' Result in file: %s\n'%name)

     def _print_tail_optimization(self,res,timer):
          self.tape.write(' ---End--- \n')
          self.tape.write(' Time %3.7f :\n'%timer)
          self.tape.write(' Message: %s\n'%res.message)
          self.tape.write(' Current energy: %f\n'%res.fun)
          self.tape.write(' Current gradient % f \n'%np.linalg.norm(res.jac,ord=np.inf))
          self.tape.write(' Number of iterations %d \n'%res.nit)
          self.sys.printcurrentgeombasis(self.tape)

     def _optprintparam(self,max_scf,rguess,maxiter=30,**kwarg):
          self.tape.write(' Initial parameters \n')
          self.tape.write(' Maximum number of optimization steps: %d\n'%maxiter)
          self.tape.write(' Tolerance in jac infinity norm (Hardcoded): %f \n'%1e-1)
          self.tape.write(' Tolerance in energy (Hardcoded): %f \n'%1e-5)
          self._printenergy(max_scf,rguess,tol=1e-8)

     def _print_head_optimization(self,name,max_scf,rguess,**kwarg):
          self.tape.write(' \n Optimization ...\n')
          self.tape.write(' Outputfiles prefix: %s'%name)
          self.tape.write(' ---Start--- \n')
          self._optprintparam(max_scf,rguess,**kwarg)


     def _energy_gradss(self,argnum,max_scf=100,max_d=300,readguess=None,name='Output.molden',output=False,order='first'):
       """This function returns the gradient of args"""
       ## For the moment it retuns a value at a time
       ## This is used only by testing functions.
       args = self._energy_args(max_scf=max_scf,max_d=max_d,log=True,printguess=None,readguess=readguess,name=name,write=output)

       if self.verbose:
          t0 = time.clock()
          self._print_head_grad(max_scf,readguess)

       ene_function = rhfenergy
       grad = grad_evaluator_algopy(ene_function,args,argnum)
       self.sys.grad = grad

       if self.verbose:
          timer = time.clock() - t0
          self._print_tail_grad(timer,grad,argnum)
       return grad

     def _singlepoint(self,max_scf=300,max_d=300,printcoef=False,name='Output.molden',output=False):
          """
          This function calculates a single point energy
          max_scf -> Maximum number of SCF cycles
          max_d ->  Maximum cycles of iterations if cannonical purification
          """
          log = True # We are not using logarithms of alphas
          eigen = True # We are using diagonalizations

       	  rguess = None
          if printcoef:
             pguess = name+'.npy'
          else:
             pguess = None


	  if self.verbose:
             self._print_head_energy(max_scf,rguess)
             t0 = time.clock()

          args = self._energy_args(max_scf=max_scf,max_d=max_d,log=True,printguess=pguess,readguess=rguess,name=name,write=output)

          # Function         
          ene = rhfenergy(*(args))
          if (ene == 99999):
             self.status = False
             print(' SCF did not converged :( !! %s\n')
          else:
             print(' SCF converged!!')
             print(' Energy: %3.7f'%ene)
             print(' Result in file: %s\n'%name)

          self.energy = ene
          if self.verbose:
             timer = time.clock() - t0
             self._print_tail_energy(timer,ene,pguess,output=output,name=name)
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
                 raise NotImplementedError("Optimization is just restricted to contraction coefficients, exponents and Gaussian centers ")
          return
          
     def _optimization(self,max_scf=100,log=True,scf=True,readguess=None,argnum=[0],taskname='Output', method='BFGS',penalize=None,**otherargs):
          print readguess
          print max_scf
	  if self.verbose:
             t0 = time.clock()
             self._print_head_optimization(taskname,max_scf,readguess,**otherargs)

          name=taskname
          rguess = None
       
          ### If initial guess
          if readguess:
             pguess = name +'.npy'
             out = False
             ene = self._singlepoint(max_scf,max_d,printcoef=pguess,name=name,output=False)
	     if ene == 99999:
                raise NameError('SCF did not converved')
             rguess= pguess

          pguess = None
          record = False

          args = self._energy_args(max_scf=max_scf,max_d=max_scf,log=log,
                                   printguess=pguess,readguess=rguess,
                                   name=name,write=record)
          ene_function = rhfenergy
          grad_fun = function_grads_algopy(ene_function,argnum)  ## This defines de gradient function of autograd
          opt = self.select_method.get(method,lambda: self._BFGS)  ## This selects the opt method

          res = opt(ene_function,grad_fun,args,argnum,log,name,**otherargs)

	  if self.verbose:
             timer = time.clock()-t0
             self._print_tail_optimization(res,timer)

          self.status = bool(res.status == 0 or res.status==1)
          self._optupdateparam(argnum,res.x)
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
        res,timer = self._optimization(max_scf,log,scf,readguess,argnum,taskname=name,**kwargs)
        return

     def gradient(self,argnum=0,max_scf=30,max_d=300,printguess=None,output=False,**kwargs):
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
        self._energy_gradss(argnum,max_scf,max_d,printguess,name,output)
        return self._energy_gradss(argnum,max_scf,max_d,printguess,name,output)

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
        
              'Grad' the energy gradient with respect to a parameter

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
                     name='../tests/testfiles/h2_sto_3g',      ## Prefix for all optput files
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
                     output=True,
                     maxiter=2)

     manager.runtask('Opt',
                     max_scf=50,
                     printcoef=False,
                     argnum=[0],
                     output=True,
                     maxiter=2)

     manager.runtask('Opt',
                     max_scf=50,
                     printcoef=False,
                     argnum=[0],
                     output=True,
                     maxiter=2)

     manager.runtask('Grad',
                     max_scf=50,
                     printcoef=False,
                     argnum=[0],
                     output=True)

     manager.end()
     return


if __name__ == "__main__":
     main()

