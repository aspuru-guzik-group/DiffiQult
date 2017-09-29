import numpy as np
import decimal
from mayavi import mlab
from mayavi.mlab import *
from tvtk.tools.visual import iterate,show
from tvtk.util.ctf import ColorTransferFunction,rescale_ctfs
from tvtk.util.gradient_editor import hsva_to_rgba
from Readmolden import Molecule

## This is for checking if a file exists or not
## Only works of python2
import os
from pathlib2 import Path

''' This module reads molden outputs from AutoChem and produces 
    Molecular orbitals and AO from it'''
# Defining general colors
black = (0,0,0)
white = (1,1,1)
gray = (0.5, 0.5, 0.5)
red = (1, 0, 0)
green = (0, 1, 0)
blue = (0, 0, 1)
color_dict = {1: (1, 1, 1), 9: (1,1,0) , 3:green, 6: (1.0,0,0), 8: (1,0,0), 7:green}
scale_dict = {1: 3.5, 9: 4.5, 7:4.5,  6:0.7, 8: 4, 3:0.5}

class Plot_system:
    ''' This class handdles contour plots and 3d plots of MOs and AO 
    (maybe in the future charges densities)
    Parameters:
	    name: filename with MOs and AOs information      | str
	    MOs: List of MOs or AOs to plot                  | list of int
    Optional:
	    ngrid: number of points to calculate in the grid | int
	    grid:  size of the gris                          | float
	    new_cube: Are we calculating again the cube file?| bool
	    nao: Are we plotting AOs or MOs?                 | bool'''
    def __init__(self,name,MOs,ngrid=40,grid = 4.5*2.0/40,new_cube=False,nao=False,density=False,ne=5):
	self.nameframe = name
	self.mol = Molecule(name)
	self.ngrid = ngrid
	self.grid = grid
	self.new_cube = new_cube
	self.nao= nao
	self.cubes = {}
        self.density =density
        self.ne = ne
	self._get_cubes(MOs)
       
	
    def _get_cubes(self,MOs):
	'''This function generates the cubes of the MOs'''
        if self.density:
            self.cubes[0] = self._getcube(0)
        else:
	    for i in MOs:
               self.cubes[i] = self._getcube(i)
	return

    def _getcube(self,MO):
	'''This function make sure that the cube file exist with the desired characteristics (included in the name of 
	the file) or if we want to calculate it again, and creates or not the file'''
        if self.density:
           namecube = self.nameframe[:-7]+'_density_'+str(MO)+'_grid'+str(self.ngrid)+'-'+str(self.grid)+'.cube'
	elif self.nao:
           namecube = self.nameframe[:-7]+'_ao_'+str(MO)+'_grid'+str(self.ngrid)+'-'+str(self.grid)+'.cube'
	else:
           namecube = self.nameframe[:-7]+'_mo_'+str(MO)+'_grid'+str(self.ngrid)+'-'+str(self.grid)+'.cube'
	if self.new_cube or not (os.path.isfile('./'+namecube)):
           self.create_cube(namecube,MO,label=self.nameframe,nao=self.nao)
        print namecube
	return namecube

    def plot_MOs(self, mol,nmo,contour=None,nao = None):
        ''' If produces countors of MOs in Mayavi 
           Originally written by Thomas Markovick '''

        natoms = len(mol.atoms)
        xarray = np.zeros((natoms, ))
        yarray = np.zeros((natoms, ))
        zarray = np.zeros((natoms, ))
	
        for i in range(natoms):
             (xarray[i], yarray[i], zarray[i]) = mol.atoms[i].xyz
             at = mlab.points3d(xarray[i], yarray[i], zarray[i],
                       scale_factor=scale_dict[float(mol.atoms[i].at)],
                       resolution=20,
                       color=color_dict[mol.atoms[i].at],
                       scale_mode='none')
        x, y, z = np.mgrid[min(xarray)-10.0:max(xarray)+10.0:100j,
                       min(yarray)-10.0:max(yarray)+10.0:100j, 
                       min(zarray)-10.0:max(zarray)+10.0:100j]

        if nao != None:
              fxyz = mol.getMOvalue(nmo,x,y,z,self.ngrid,nao=nao,ne=self.ne)
        else:
              fxyz = mol.getMOvalue(nmo,x,y,z,self.ngrid)
        fxyz = np.abs(fxyz)
        src = mlab.pipeline.scalar_field(x,y,z,fxyz)
        mlab.pipeline.iso_surface(src, contours=[-0.1,0.1], opacity=0.6)
        mlab.show()
    
    def create_cube(self,tape,nmo,label=None,nao=False):
         '''This function creates a cube file from the info in name on tape
         for the nmo number nmbo, if nao provided is a nao
         Parameters:
       	    tape: name of the cube file      | str
	    nmo: number of AO or MO to print | int
         optional:
	    label: Name of the molden file | str
	    nao: Are you printing AO       | bool'''
	 mol = self.mol
         origin = np.zeros(3) + self.grid*self.ngrid/2.0
         file = open(tape,"w")
     
         file.write(str(label)+'\n')
         file.write(str(nmo)+'\n')
         for i in mol.atoms:
             file.write("{:3d}    {:2.6f}     {:2.6f}    {:2.6f}    {:2.6f}\n".format(i.at,0.0,i.xyz[0],i.xyz[1],i.xyz[2]))
         for ix in np.linspace(0,self.ngrid*self.grid,self.ngrid)-origin[0]:
             for iy in np.linspace(0,self.ngrid*self.grid,self.ngrid)-origin[1]:
                for cont, iz in enumerate(np.linspace(0,self.ngrid*self.grid,self.ngrid)-origin[2]):
                   if self.density:
                   	fxyz = mol.getDensity(ix,iy,iz,5)[0][0][0]
                   else:
                   	fxyz = mol.getMOvalue(nmo,ix,iy,iz,1,nao=nao)[0][0][0]
                   file.write('{:4.5E}'.format(float(fxyz)))
                   file.write(' ')
                   if (cont%6 == 5):
                      file.write('\n')
                file.write('\n')
         file.close()
         return
    
    def plot_density(self,MO,bgcolor=white,nfig=1):
	'''This function defines a mlab object, calls frame density to make a plot
	and displays it'''
        mlab.figure(nfig, bgcolor=bgcolor)#, size=(500, 500))
        mlab.clf()
	self.frame_density(self.cubes[MO])
        mlab.show()
        name = str(MO)+'.png'
        mlab.savefig(name,size=(500,500))


    def read_data(self,MO):
	    return self. _read_density(self.cubes[MO])

    def _read_density(self,name):
        '''This function read the densisty from a cube file
        Parameters:
               name: name of the cubefile
        Output:
               data: tensor of rank 3 with the cube
               min_val: minimum value of the array
               max_val maximum value of the array'''
        natoms = len(self.mol.atoms)
        str = ' '.join(file(name).readlines()[natoms+2:])
        data = np.fromstring(str, sep=' ')
        data.shape = (self.ngrid, self.ngrid, self.ngrid) 
        min_val = np.abs(data).min()
        max_val = np.abs(data).max()
	return data,min_val,max_val

    def frame_density(self,name,data=None,min_val=None,max_val=None,density=False,label=None):
        '''Plot AO and MO density 3D plots based in an script of 
        Gael Varoquaux <gael.varoquaux@normalesup.org>'''
	mol = self.mol
        natoms = len(mol.atoms)
        xarray = np.zeros((natoms, ))
        yarray = np.zeros((natoms, ))
        zarray = np.zeros((natoms, ))
        origin = grid*ngrid/2.0

        # Ploting atoms
        for i in range(natoms):
           ## Rescaling and shifting the atoms position
           (xarray[i], yarray[i], zarray[i]) = mol.atoms[i].xyz/grid+ngrid/2.0 
           print ('atoms',xarray[i], yarray[i], zarray[i],mol.atoms[i].at,scale_dict[mol.atoms[i].at]) 
           at = mlab.points3d(xarray[i], yarray[i], zarray[i],
                          scale_factor=scale_dict[float(mol.atoms[i].at)],
                          resolution=40,
                          color=color_dict[mol.atoms[i].at],
                          scale_mode='none')

        # Ploting bonds (for now all the atoms are connected to the center).
        for i in range(1,natoms):
             mlab.plot3d([xarray[0],xarray[i]], [yarray[0],yarray[i]], [zarray[0],zarray[i]], [2, 1],
                     tube_radius=1.0, colormap='Greys')

        ## _Volume
        print type(data)
	if type(data) != np.ndarray:
            data,min_val,max_val = self._read_density(name)
        print min_val,max_val 

        source = mlab.pipeline.scalar_field(data)
        vol_pos = mlab.pipeline.volume(source, vmin=min_val + 1e-15*(min_val-max_val),vmax=max_val)
        #color_change_vol(vol_pos,min_val,max_val,(0.0,0.0,1.0))
        #surf_neg = mlab.pipeline.iso_surface(source, contours=[-0.1,0.1], opacity=0.6)


	data = np.negative(data)
        source = mlab.pipeline.scalar_field(data)
        vol_neg = mlab.pipeline.volume(source, vmin=min_val + 1e-15*(min_val-max_val),vmax=max_val)
        #color_change_vol(vol_neg,min_val,max_val,(1.0,0.0,0.0))
        #surf_neg = mlab.pipeline.iso_surface(source, contours=[-0.1,0.1], opacity=0.6)
        if label != None:
            mlab.title(label,color=(0,0,0),size=0.8)

	'''# Just reference points of the box
        (xarray[0], yarray[0], zarray[0]) = np.zeros(3)
        at = mlab.points3d(xarray[0], yarray[i], zarray[i],
                          scale_factor=1*scale_dict[float(mol.atoms[i].at)],
                          resolution=20,
                          color=color_dict[mol.atoms[i].at],
                          scale_mode='none')
        (xarray[0], yarray[0], zarray[0]) = np.zeros(3)+self.ngrid
        #print (xarray[0], yarray[0], zarray[0]) 
        at = mlab.points3d(xarray[0], yarray[i], zarray[i],
                          scale_factor=2*scale_dict[float(mol.atoms[i].at)],
                          resolution=20,
                          color=(0,0,1),
                          scale_mode='none')'''
        return


class Plot_systems():
	def __init__(self,frames,MOs,aos=False,new_cube=False,ngrid=40,grid=1.5*2/40,surface=False,density=False,labels=None,ne=5):
	    self.psyss = []
            self.labels = labels
            for i,nameframe in enumerate(frames):
		print i,nameframe
	        self.psyss.append(Plot_system(nameframe,MOs,ngrid,grid,nao=aos,new_cube=new_cube,density=density,ne=ne))

	def plotMO(self,MO,nfig=1,bgcolor=white):
	    '''This function defines a mlab object, calls frame density to make a plot
	    and displays it'''
	    datas = []
	    mins = []
	    maxs = []
	    for sys in self.psyss:
	    	data,min_val,max_val = sys.read_data(MO)
                datas.append(data)
                mins.append(min_val)
                maxs.append(max_val)
            print mins,maxs

            #datas[1] = np.negative(datas[1])
            all_min = min(mins)
            all_max = max(maxs)
            all_max *= 1e-2
	    for i,psys in enumerate(self.psyss):
                mlab.figure(nfig, bgcolor=bgcolor)#, size=(500, 500))
                mlab.clf()
	        psys.frame_density(psys.cubes[MO],data=datas[i],min_val=all_min,max_val=all_max,label=self.labels[i])
                mlab.view(azimuth=0,elevation=90)
                mlab.show()
                

def color_change_vol(vol,min_val,max_val,rgb):
        '''r = rgb[0]
        g = rgb[1]
        b = rgb[2]
        ctf = ColorTransferFunction()
        ctf.add_rgb_point(min_val, r,g, b)
        ctf.add_rgb_point(max_val, r,g, b)
        vol._volume_property.set_color(ctf)
        vol._ctf = ctf
        vol.update_ctf = True
        return '''
        hue_range=[2.0/3.0, 0.0]
        sat_range=[1.0, 1.0]
        val_range=[1.0, 1.0]
        n=50
        ds = max_val-min_val
        dhue = hue_range[1] - hue_range[0]
        dsat = sat_range[1] - sat_range[0]
        dval = val_range[1] - val_range[0]
        ctf = ColorTransferFunction()
        for i in range(n+1):
            x = 0.5*(1.0  + np.cos((n-i)*np.pi/n)) 
            h = hue_range[0] + dhue*x
            s = sat_range[0] + dsat*x
            v = val_range[0] + dval*x
            r, g, b, a = [np.sqrt(c) for c in hsva_to_rgba(h, s, v, 1.0)]
            #r *= rgb[0]
            #g *= rgb[1]
            #b *= rgb[2]
            ctf.add_rgb_point(min_val+x*ds, r, g, b)
        vol._volume_property.set_color(ctf)
        vol._ctf = ctf
        vol.update_ctf = True
        return




if __name__ == '__main__':
	''' ### Example of ploting one frame an Mo at a time
        for nameframe in frames[0:1]:
            psys = Plot_system(nameframe,[4],ngrid,grid,nao=ao)
	    psys.plot_density(4)

	'''
	## Parameters
        ngrid = 40
        grid = 4.5*2.0/ngrid
	new_cube = True
	new_cube = False
	ao = True
	ao = False
        surface = False
        density = True

	## Name of frames
	frames = ['./ch4/ch4-a-later-z-sto-2g-BFGS_step_27.molden','./ch4/ch4-sto-2g-BFGS_step_0.molden']
	frames = [ './hf/hf-sto-2g-BFGS_step_0.molden', './hf/hf-sto-2g-BFGS_step_40.molden']
	labels = [ '2G-STO','2G-STO width and position']


	frames = [ 'h2o_3_cluster_-sto-2-BFGS_step_0.molden']

        frames = ['./ch4/ch4-a-and-z-sto-2g-BFGS_step_29.molden', './ch4/ch4-sto-3g-BFGS_step_0.molden', './ch4/ch4-sto-3g-BFGS_step_44.molden', './ch4/ch4-a-later-z-sto-3g-BFGS_step_9.molden', './ch4/ch4-sto-6g-BFGS_step_0.molden']
        labels = ['STO-3G','STO-3G optimization of width','STO-3G width and position',
         'STO-3G','STO-3G optimization of width','STO-3G width and position',
         'STO-6G']

	#frames = [ './hf/hf-sto-2g-BFGS_step_0.molden', './hf/hf-sto-2g-BFGS_step_40.molden', './hf/hf-sto-6g-BFGS_step_0.molden'
	#frames = [ './h2o/h2o-sto-2g-BFGS_step_0.molden', './h2o/h2o-a-later-z-sto-2g-BFGS_step_29.molden', './h2o/h2o-sto-6g-BFGS_step_0.molden']
	#labels = [ '2G-STO','2G-STO width and position','6']
        frames=['./nh3/nh3-a-later-z-sto-2g-BFGS_step_29.molden', './nh3/nh3-sto-3g-BFGS_step_0.molden', './nh3/nh3-sto-3g-BFGS_step_51.molden', './nh3/nh3-a-later-z-sto-3g-BFGS_step_9.molden', './nh3/nh3-sto-6g-BFGS_step_0.molden']


        nmos = range(0,4)
        systems = Plot_systems(frames,nmos,aos=ao,ngrid=ngrid,grid=grid,new_cube=new_cube,surface=surface,density=density,labels=labels,ne=5*3)
        for i in range(1):
	    systems.plotMO(i)
