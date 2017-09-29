import imageio
import os
import os.path


def makemovie(nframes, frame_name, movie_name ='movie.gif'):
    ''' This python function builds a a movie from a enumerated
        series of images with the name: frame_name#.png:
            frames: Numbers of frames
            frame_name: The name of frame.
            movie_name: The name of the gif file that it is going to generate
    '''
    images = []
    for i in range(nframes):
        images.append(imageio.imread(frame_name+str(i)+".png"))
    imageio.mimsave(movie_name, images, duration=1.0)
    return

def get_lastfile(molecule,filename):
    ''' This function returns the last step of the optimization path inside of a folder called
     ./molecule/ and with the prefix filename
    Parameters:
          molecule: Name of the folder, one level below | str
          filename: Prefix of the file name             | str
    Output:
         fistname: Name of the first molden file of the opt | str   
         lastname: Name of the last molden file of the opt  | str  
         index:    Returns the number of optimization steps or -1 if it can find a single step | int'''
    index=-1
    for file in os.listdir(molecule):
        if file.startswith(filename+'-BFGS_step_') and file.endswith('.molden'):
            trim = file.replace(filename+'-BFGS_step_', "")
            trim =trim.replace('.molden', "")
            if int(trim) > index:
                index = int(trim)
    lastfile = "./"+molecule+'/'+ filename+'-BFGS_step_'+str(index)+'.molden'
    firstfile = "./"+molecule+'/'+ filename+'-BFGS_step_'+str(0)+'.molden'
    return firstfile,lastfile,index
