import scipy as sc
from scipy import special
import numpy as np
import itertools


def vec_tomatrix(x,n):
    reverse_x = x
    for i in range(n):
       reverse_x -= n-i
       if reverse_x < 0:
          break
    j = n+reverse_x
    return i,j
    
            
def matrix_tovector(i,j,n):
    '''This function returns the index of the vector 
    for a given i,j element of a matrix
    Note: j < i, [i,j] = [j,i] '''
    
    return int(i*n-i*(i+1)/2+j)
       

def eri_index(i,j,k,l,n):
    '''This function returns the index of a vector, that maps to
    a matrix with indexes, x,y
    where x and y are indexes that maps to the matrices i,j and k,l.
    '''
    ##  We are checking the highest number
    if (k >= l):
       kk = l
       ll = k
    else:
       kk = k
       ll = l

    if (i >= j):
       ii = j
       jj = i
    else:
       ii = i
       jj = j
    x = matrix_tovector(ii,jj,n)
    y = matrix_tovector(kk,ll,n)
    new_n = n*(n + 1)/2 ## The shape of the new matrices i,j and k,l
    if x >= y:
       elem = matrix_tovector(y,x,new_n)
    else:
       elem = matrix_tovector(x,y,new_n)
    return int(elem)

def euclidean_norm2(tensor):
    return np.sum(np.square(tensor),axis=0)

def euclidean_norm(tensor):
    return np.sqrt(euclidean_norm2(tensor))
    #return np.sqrt(tensor[0]*tensor[0]+tensor[1]*tensor[1]+tensor[2]*tensor[2])
    #return np.linalg.norm(tensor)

def factorial(n):
    tmp = 1
    for i in range(n):
        tmp *= (i+1)
    return tmp

def binomial(n,k):
    return math.factorial(n)/(math.factorial(n-k)*math.factorial(k))

def grad_incompletegammaf(ans,t,m):
    '''Gradient of boys function '''
    def gradient_element(g):
        eps = 0.5e-12
        t_new = max(t,1e-12)
        return g*(-(m+0.5)*ans + 0.5*np.exp(-t_new))/t_new
    return gradient_element

def incompletegammaf(t,n):
    '''This function returns the inclomplete gamma function
    based on Numerical Recipies Book and pag 7 of Methods in 
    Computational Physics'''
    t = max(t,1e-12)
    return getgammp(n+0.5,t)*pow(t,-n-0.5)*0.5#*pow(2/np.pi,0.5)
    #return np.multiply(getgammp(n+0.5,t),pow(t,-n-0.5))*0.5#*pow(2/np.pi,0.5)
    #return getgammp(n+0.5,t)

def getgammp(a,x):
    ''' Returns gamma incomplete function times gamma(a), Numerical Recipies p. 218 '''
    if x < 0:
       print 'Eri or Nuclear integrals, gammainc: Not valid value of t '
       exit()
    if (x < a + 1.0):
       #print 'gser'
       return gser(a,x)
    else:
       #print 'gcf'
       return gcf(a,x)

def gammaln(a):
    ''' Returns the value of ln(gamma(a), Numerical Recipes p. 214'''
    coef= [76.18009172947146,-86.50532032941677,24.01409824083091,-1.231739572450155,0.1208650973866179e-2,-0.5395239384953E-5]
    y=x=a
    tmp = x+5.5
    tmp = tmp - (x+0.5)*np.log(tmp)
    ser = 1.000000000190015
    for i in coef:
       y = y + 1.0
       ser = ser + i/y
    return -tmp+np.log(2.5066282746310005*ser/x)

def gser(a,x):
   eps = 3.0E-12
   itmax = 10000
   ap = np.copy(a)
   de = summ = 1.0/a
   for i in range(itmax):
      ap = ap + 1.0
      de = de*x/ap
      summ = summ + de
      if (de < summ*eps):
        gln = gammaln(a)
        #return (summ*np.exp(-x+a*np.log(x)-gln))*np.exp(gln)
        return np.multiply(summ,np.multiply(np.exp(np.add(np.add(-x,np.multiply(a,np.log(x))),-gln)),np.exp(gln)))
   print 'Eri or Nuclear integrals, gammainc: Series representation does not converge',a
   exit()

def gcf(a,x):
   eps = 3.0E-7
   fpmin = 1.0E-30
   itmax = 10000
   b = x+1.0-a
   c = 1.0/fpmin
   d = 1.0/b
   h = d
   for i in range(1,itmax):
       an = -i*(i-a)
       b = 2.0 + b
       d = an*d+b
       if (abs(d) < fpmin):
          d = fpmin    
       c = b +an/c
       if (abs(c) <fpmin):
          c = fpmin    
       d = 1.0/d
       de = d*c
       h = h*de
       if (abs(de-1.0) < eps):
          gln = gammaln(a)
          return (1.0-np.exp(-x+a*np.log(x)-gln)*h)*np.exp(gln)
   print 'Eri or Nuclear integrals, gammainc: Inc. frac. representation does not converge'
   exit()


def getfsh(t,m):
   eps = 1.0E-15
   nmax = 1000
   summation = 0.0
   for i in range(1,nmax):
      new = pow(t,i)/sc.special.gamma(m+i+1.5)
      summation = summation + new
      if (new < eps):
        return summation*0.5/sc.special.gamma(m+0.5)*np.exp(-t)

def gammainc(t,n):
   ''' Savitt method and Davidson's '''
   if (t<13):
     ts = t-0.02
     new = getfsh(ts,0)
     summation = new
     for i in range(1,7):
         fact = ts
         new*= getfsh(ts,i+1)/i
         summation = summation + new
     return summation
      
def printmatrix(matrix,tape):
    np.set_printoptions(precision=7)
    line = '  '
    for i, col in enumerate(matrix):
       line += str(i) + '     '
    tape.write(line+'\n')
    for i, col in enumerate(matrix):
       line = str(i) + ' '
       for j, elem in enumerate(col):
           line += str(elem)+' '
       tape.write(line+'\n')
    return
     

def grad_eigen(ans,x):
    """Gradient for eigenvalues and vectors of a symmetric matrix with repeated eigenvalues."""
    N = x.shape[-1]
    w, v = ans              # Eigenvalues, eigenvectors.
    dot = np.dot if x.ndim == 2 else partial(np.einsum, '...ij,...jk->...ik')
    T = lambda x: np.swapaxes(x, -1, -2)
    print 'x',x
    print 'lam',w
    print 'vec',v
    def eigh_grad(g):
        tool = 1.0e-5
        wg, vg = g          # Gradient w.r.t. eigenvalues, eigenvectors.
        w_repeated = np.repeat(w[..., np.newaxis], N, axis=-1)
        F = T(w_repeated) - w_repeated 
        for i in xrange(F.shape[0]):
           for j in xrange(F.shape[1]):
              if F[i,j] < tool:
                 F[i,j] = 0.0
              else:
                 F[i,j] = 1.0/F[i,j]
        print F
        return dot(v * wg[..., np.newaxis, :] + dot(v, F * dot(T(v), vg)), T(v))
    return eigh_grad

def eigensolver(X):
    return np.linalg.eigh(X)
