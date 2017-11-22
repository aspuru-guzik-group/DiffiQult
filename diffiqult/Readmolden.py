import numpy as np
import decimal
## This is for checking if a file exists or not
## Only works of python2
import os

''' This module reads molden outputs from AutoChem and produces 
    Molecular orbitals and AO from it'''

class Molecule:
    ''' This class have all the information about MOs, AOs, their values in
        a given point, energies and.at general info of a molecule. '''
    def __init__(self,name):
        #self.name = name
        self.atoms=[]
        self.gauss=[]
        self.getcoordinates(name)
        self.gettotene(name)
        ## Getting center of mass
        self.xyz = self.getcenter()
        ## Getting exp and coef of gaussians
        self.getgto(name)
        self._normalizationgauss()

        self.nMOs = len(self.gauss)
        ## Getting coefficient matrix
        self.C=np.zeros([self.nMOs,self.nMOs],dtype='float64')
        self.getmosc(name)
        ## Getting energy of orbitals
        self.e=np.zeros([self.nMOs],dtype='float64')
        self.nbasis = len(self.gauss)
        
        # It does not contain the energy self.getene(name)
        #print self.C
        #self.printinfo()
        
    def gettotene(self,name):
        '''This module finds the line where
        we'll collect the total energy '''
        tape = open(name,"r")
        label = False
        lines=tape.readlines()
        for i,line in enumerate(lines):
            if label:
               self.tote = float(lines[i+2].split(' ')[1])
               return
            if (line.replace(" ","")[0:8]=='[Energy]'):
               label = True
        tape.close()

    def getene(self,name):
        '''This module finds the line where
        we'll collect the energy of MOE'''
        tape = open(name,"r")
        label = False
        lines=tape.readlines()
        for i,line in enumerate(lines):
            if label:
                self.tote = lines[i+2].split(' ')[1]
                return
            if (line.replace(" ","")[0:5]=='[MOE]'):
                label = True
        tape.close()

    def gete(self,index,lines):
        '''This module colllects the energy of MOE'''
        for i in range(index,index+len(self.gauss)):
           line = lines[i]
           if (line.replace(" ","")[0]=='['):
              return
           else:
              val = line.split(' ')[1]
              self.e[i-index] = value
        return

    def getmosc(self,name):
        tape = open(name,"r")
        label = False
        lines=tape.readlines()
        for i,line in enumerate(lines):
            if label:
               self.getc(i+2,lines)
               return
            if (line.replace(" ","")[0:4]=='[CM]'):
               label = True
        tape.close()

    def getc(self,index,lines):
        for i in range(index,index+len(self.gauss)):
           line = lines[i]
           if (line.replace(" ","")[0]=='['):
              return
           else:
              row = line.split(' ')
              for j,value in enumerate(row[1:self.nMOs+1]):
                 self.C[i-index,j] = value
        return

    def getgto(self,name):
        tape = open(name,"r")
        label = False
        lines=tape.readlines()
        for i,line in enumerate(lines):
            if label:
               self.getgauss(i,lines)
               break
            if (line.replace(" ","")[0:5]=='[GTO]'):
               label = True
        tape.close()
        return
    
    def getgauss(self,index,lines):
        i = index
        for j in range(len(self.gauss)):
           line = lines[i]
           if (line.replace(" ","")[0]=='['):
              return
           else:
              label = line.split(' ')[2]
              if (self.gauss[j].label == label):
                    i += 1
                    line = lines[i]
                    spdf = line.split(' ')
                    self.gauss[j].contr = int(spdf[4])
                    self.gauss[j].l[0] = int(spdf[6])
                    self.gauss[j].l[1] = int(spdf[7])
                    self.gauss[j].l[2] = int(spdf[8])
                    for jj in range(self.gauss[j].contr):
                        i += 1
                        line = lines[i]
                        tmp = Primitive()
                        tmp.exp = float(line.split(' ')[2])
                        tmp.coef = float(line.split(' ')[3])
                        self.gauss[j].prim.append(tmp)
                    i += 1
                    #if (self.gauss[j].contr == 1):
                    #    i += 1
              if len(lines[i].strip()) == 0:
                 i += 1
            
        return

    def getcoordinates(self,name):
        tape = open(name,"r")
        label = False
        for i,line in enumerate(tape):
            if label:
               if (line.replace(" ","")[0]=='['):
                  return
               elif line[0] != 'X':
                  tmp = Atom()
                  #print line.split(' ')
                  tmp.sym,tmp.label,tmp.at,tmp.xyz[0],tmp.xyz[1],tmp.xyz[2] = line.split(' ')
                  tmp.at = int(tmp.at)
                  tmp.xyz[0] = float(tmp.xyz[0])
                  tmp.xyz[1] = float(tmp.xyz[1])
                  tmp.xyz[2] = float(tmp.xyz[2])
                  tmp.xyz = np.array(tmp.xyz)
                  self.atoms.append(tmp)
               elif (line[0] == 'X'):
                  tmp = Gauss()
                  trash1,tmp.label,trash2,tmp.xyz[0],tmp.xyz[1],tmp.xyz[2] = line.split(' ')
                  tmp.xyz[0] = float(tmp.xyz[0])
                  tmp.xyz[1] = float(tmp.xyz[1])
                  tmp.xyz[2] = float(tmp.xyz[2])
                  tmp.xyz = np.array(tmp.xyz)
                  self.gauss.append(tmp)
            if (line.replace(" ","")[-8:-1]=='[Atoms]'):
               label = True
        tape.close()
        return
   
    def getcenter(self):
        xyz = np.zeros(3)
        for i in self.atoms:
            xyz += i.xyz  
        xyz = xyz/len(self.atoms)
        return xyz

    def geteDensityvalue(self,nMO,x, y, z, ngrid):
        nao=False
        #for i in self.


    def getDensity(self,x,y,z,ngrid,ne=2):
        res = np.zeros((ngrid,ngrid,ngrid))
        for i in range(ne):  
           res = np.add(res,np.power(self.getMOvalue(i,x, y, z,ngrid),2.0))
        return 2.0*res
      
    def getMOvalue(self,nMO,x, y, z, ngrid,nao=False):
        fxyz = np.zeros((ngrid, ngrid, ngrid))
        if nao:
            fxyz += self.getAOvalue(nMO,x, y, z)
        else:
           for i in range(self.nMOs):
              fxyz += self.C[i,nMO]*self.getAOvalue(i,x, y, z)
        return fxyz

    def getAOvalue(self,nAO,x, y, z):
        ao = self.gauss[nAO]
        #for pao in ao.prim:
             #print (pao.exp, pao.coef)
        exp = 0.0
        value = 0.0
        exp -= np.subtract(x,ao.xyz[0])*np.subtract(x,ao.xyz[0])
        exp -= np.subtract(y,ao.xyz[1])*np.subtract(y,ao.xyz[1])
        exp -= np.subtract(z,ao.xyz[2])*np.subtract(z,ao.xyz[2])
        for pao in ao.prim:
            value += np.multiply(pao.coef,np.exp(np.multiply(pao.exp,exp)))
            #print('p',pao.coef,pao.exp)
        if ao.l[0] == 1:
            value *= ao.xyz[0]-x
        if ao.l[1] == 1:
            value *= ao.xyz[1]-y
        if ao.l[2] == 1:
            value *= ao.xyz[2]-z
            
        return value
        
    def getAOcontibution(self,nMO,nAO,xyz):
        return self.C[nMO,nAO]*self.getvalue(i,xyz)


    def _normalizationgauss(self):
        for gauss in self.gauss:
            gauss.norm()

   
class Atom:
    def __init__(self):
        self.xyz = [0.,0.,0.]
        self.sym = "XX" 
        self.label = 0
        self.at = None
              
class Gauss:
    def __init__(self):
        self.xyz = [0.,0.,0.]
        self.l = [0.,0.,0.]
        self.label = 0
        self.prim = []
        self.contr = 0
      
    def norm(self):
        l_large = self.l[0] + self.l[1] + self.l[2]
        div = 1.0
        for m in self.l:
         if (m>0):
           for k in range(1,2*m,2):
              div = div*k
        div = div/pow(2,l_large)
        div = div*pow(np.pi,1.5)
        for pao in self.prim:
              pao.coef = pao.coef*self._normalization_primitive(pao.exp,self.l,l_large)
        tmp = 0.0
        for paoi in self.prim:
           for paoj in self.prim:
               tmp = tmp+ paoj.coef*paoi.coef/np.power(paoi.exp+paoj.exp,l_large+1.5)
        tmp = np.sqrt(tmp*div)
        for paoi in self.prim:
            paoi.coef = paoi.coef/tmp
        return 

        
    def _normalization_primitive(self,alpha,l,l_large):
        factor = []
        power = l_large/2.0
        div = np.float64(1.0)
        for m in l:
          if (m>0):
            for k in range(1,m+1,2):
                div = div*2.0*k
                div = np.sqrt(div)
        factor = np.power((2.0/np.pi),0.75)*div*pow(2.0,power)
        power = power + 0.75 
        return factor*np.power(alpha,power)


def normalization(alpha,c,xyz,l,list_contr,dtype=np.float64(1.0)):
    contr = 0
    coef = algopy.zeros(len(alpha),dtype=dtype)
    
    for ib, ci in enumerate(list_contr):
       div = 1.0

class Primitive:
    def __init__(self):
        self.exp = 0.
        self.coef = 0.
        
def readmos(name):
    return Molecule(name)


if __name__ == '__main__':
        frames = ['./Data-contracted-ac-z-ac-z/h2o/h2o-sto-2g-task-1-BFGS_step_0.molden','./Data-primitive-a-z-a-z/h2o/h2o-sto-2g-task-1-BFGS_step_0.molden']
        sys = Molecule(frames[0])
        sys = Molecule(frames[1])

