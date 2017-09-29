import numpy as np
import decimal
## This is for checking if a file exists or not
## Only works of python2
import os

''' This module reads molden outputs from AutoChem and produces 
    Molecular orbitals and AO from it'''
# Defining general colors
black = (0,0,0)
white = (1,1,1)
gray = (0.5, 0.5, 0.5)
red = (1, 0, 0)
green = (0, 1, 0)
blue = (0, 0, 1)
color_dict = {1: (1, 1, 1), 9: red, 3:green, 6: (1.0,0,0), 8: (1,0,0), 7:green}
scale_dict = {1: 0.5, 9: 0.5, 7:0.5,  6:0.7, 8: 0.5, 3:0.5}

class Molecule:
    ''' This class have all the information about MOs, AOs, their values in
        a given point, energies and.at general info of a molecule. '''
    def __init__(self,name):
        self.atoms=[]
        self.gauss=[]
        self.getcoordinates(name)
        self.gettotene(name)
        ## Getting center of mass
        self.xyz = self.getcenter()
        ## Getting exp and coef of gaussians
        self.getgto(name)
        self.nMOs = len(self.gauss)
        ## Getting coefficient matrix
        self.C=np.zeros([self.nMOs,self.nMOs],dtype='float64')
        self.getmosc(name)
        ## Getting energy of orbitals
        self.e=np.zeros([self.nMOs],dtype='float64')
        
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
               return
            if (line.replace(" ","")[0:5]=='[GTO]'):
               label = True
        tape.close()
    
    def getgauss(self,index,lines):
        list_contr = []
        for i in range(index,len(lines),4):
           line = lines[i]
           if (line.replace(" ","")[0]=='['):
              return
           else:
[GTO]
  4  0
 p   1 1.0 0 0 0
  130.70932 14.1129382769
  23.808861 3.80573157866
  6.4436083 0.445048197405
              label = line.split(' ')[2]
              #label,trash,l = line.split(' ') #It will contain l for p orbitals (0,1,2) x=0 y=1, z=2
              cont,lx,lx,lz = (int(line.split(' ')[5]))
              for j in range(len(self.gauss)):
                 if (self.gauss[j].label == label):
                    line = lines[i+1]
                    spdf = line.split(' ')
                    #print 'spdf',spdf
                    #if spdf[1] == 'p' or spdf[1] == 's':
                    self.gauss[j].l[0] = int(spdf[6])
                    self.gauss[j].l[1] = int(spdf[7])
                    self.gauss[j].l[2] = int(spdf[8])
                    #printpipline volume 'self',self.gauss[j].l

                    #else:
                       #print (spdf)
                     #  print ('We can not handdle beyond p orbitals')
                     #  exit()
                    line = lines[i+2]
                    self.gauss[j].exp = float(line.split(' ')[2])
                    self.gauss[j].coef = float(line.split(' ')[3])

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
            if (line.replace(" ","")[0:7]=='[Atoms]'):
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
        return res
      
    def getMOvalue(self,nMO,x, y, z, ngrid,nao=False):
        #print 'inside mo',self.C
        fxyz = np.zeros((ngrid, ngrid, ngrid))
        if nao:
            fxyz += self.getAOvalue(nMO,x, y, z)
        else:
           for i in range(self.nMOs):
              fxyz += self.C[i,nMO]*self.getAOvalue(i,x, y, z)
        return fxyz

    def getAOvalue(self,nAO,x, y, z):
        ao = self.gauss[nAO]
        exp = 0.0
        value = 1.0
        exp -= np.subtract(x,ao.xyz[0])*np.subtract(x,ao.xyz[0])
        exp -= np.subtract(y,ao.xyz[1])*np.subtract(y,ao.xyz[1])
        exp -= np.subtract(z,ao.xyz[2])*np.subtract(z,ao.xyz[2])
        if ao.l[0] == 1:
            value *= ao.xyz[0]-x
        if ao.l[1] == 1:
            value *= ao.xyz[1]-y
        if ao.l[2] == 1:
            value *= ao.xyz[2]-z
        value *= np.multiply(ao.coef,np.exp(np.multiply(ao.exp,exp)))
        return value
        
    def getAOcontibution(self,nMO,nAO,xyz):
        return self.C[nMO,nAO]*self.getvalue(i,xyz)


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
        self.exp = 0.
        self.coef = 0.

def readmos(name):
    return Molecule(name)


if __name__ == '__main__':
        frames = ['./h2o/h2o-a-later-z-sto-2g-BFGS_step_27.molden','./h2o/h2o-sto-2g-BFGS_step_0.molden']
        sys = Molecule(frames[0])

