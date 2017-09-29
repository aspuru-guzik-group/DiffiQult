import autograd.numpy as np
from autograd import grad
from autograd import jacobian
from autograd import elementwise_grad
from autograd import scipy
import autograd.scipy as sc
import Tools
from Tools import *

## For now let's treat all stuff as S orbitals

def eriss_grad(a,A,b,B,c,C,d,D):
    ''' This version does not take into account at all the angula momentum '''
    eps = 1e-5
    gamma = a+b
    pab = np.divide(np.add(np.multiply(a,np.array(A)),np.multiply(b,np.array(B))),gamma)
    #fab = coefa*coefb
    ab = euclidean_norm2(np.subtract(np.array(A),np.array(B)))/gamma
    #kab = fab*np.exp(-1.0*a*b*ab)
    kab = np.exp(-1.0*a*b*ab)

    nu = c+d
    qcd = np.divide(np.add(np.multiply(c,np.array(C)),np.multiply(d,np.array(D))),nu)
    #fcd = coefc*coefd
    cd = euclidean_norm2(np.subtract(np.array(C),np.array(D)))/nu
    #kcd = fcd*np.exp(-1.0*c*d*cd)
    kcd = np.exp(-1.0*c*d*cd)

    rho = nu*gamma/(nu+gamma)
    t = rho*euclidean_norm2(np.subtract(np.array(pab),np.array(qcd)))
    fact = t - t/(nu*gamma)
    fact += 2.0*t*euclidean_norm(np.subtract(np.divide(np.array(A),gamma),np.divide(np.array(pab),gamma)))
    
    dt = 2.0/np.sqrt(np.pi)*math.exp(-1.0*t*t)
    return dt #1/(dt*t)


def gradient(i,j,k,l,alpha,xyz,nbasis,G):
    eps = 1e-5
    res = np.zeros(nbasis, dtype='float64')



    ## dt/dg
    gamma = alpha[i]+alpha[j]
    nu = alpha[k]+alpha[l]
    pab = np.divide(np.add(np.multiply(alpha[i],xyz[i]),np.multiply(alpha[j],xyz[j])),gamma)
    qcd = np.divide(np.add(np.multiply(alpha[k],xyz[k]),np.multiply(alpha[l],xyz[l])),nu)
    nu_gamma = 1.0/(nu+gamma)
    rho = nu*gamma*nu_gamma
    pq = euclidean_norm(np.subtract(pab,qcd))
    t = rho*pq*pq
    if (t < eps):
       return res  
    #tdg_dt = 2.0*pow(t,0.5)*np.exp(-t) - 0.5*G
    tdg_dt = 0.5*np.exp(-t) - 0.5*G

    ## ab's
    
    ap = np.subtract(xyz[i],pab)
    da_dt_t = (1.0+2.0*(ap[0]+ap[1]+ap[2])/pq)/gamma - nu_gamma 
    res[i] += (tdg_dt*da_dt_t)


    ap = np.subtract(xyz[j],pab)
    da_dt_t = (1.0+2.0*(ap[0]+ap[1]+ap[2])/pq)/gamma - nu_gamma 
    res[j] += (tdg_dt*da_dt_t)


    ap = np.subtract(xyz[k],qcd)
    da_dt_t = (1.0-2.0*(ap[0]+ap[1]+ap[2])/pq)/nu - nu_gamma 
    res[k] += (tdg_dt*da_dt_t)


    ap = np.subtract(xyz[l],qcd)
    da_dt_t = (1.0-2.0*(ap[0]+ap[1]+ap[2])/pq)/nu - nu_gamma 
    res[l] += (tdg_dt*da_dt_t)

    return res


def tensor_eris(ans,alpha,xyz,nbasis):
    res = np.zeros((nbasis,nbasis,nbasis,nbasis,nbasis), dtype='float64')
    for i in range(nbasis):
       for j in range(nbasis):
          for k in range(nbasis):
              for l in range(nbasis):
                 res[i,j,k,l] = gradient2(i,j,k,l,alpha,xyz,nbasis,ans[i,j,k,l])
    return np.reshape(np.array(res),(nbasis,nbasis,nbasis,nbasis,nbasis))
                     
    
def gradient_g(ans,alpha,xyz,nbasis):
    def gradient_element(g):
        x_w_g = tensor_eris(ans,alpha,xyz,nbasis)
        res =  np.tensordot(x_w_g,g,axes=([0,1,2,3],[0,1,2,3]))
        return res
    return gradient_element


def gradient2(i,j,k,l,alpha,xyz,nbasis,G):
    eps = 1e-5
    res = np.zeros(nbasis, dtype='float64')

    ## dt/dg
    gamma = alpha[i]+alpha[j]
    nu = alpha[k]+alpha[l]
    pab = np.divide(np.add(np.multiply(alpha[i],xyz[i]),np.multiply(alpha[j],xyz[j])),gamma)
    qcd = np.divide(np.add(np.multiply(alpha[k],xyz[k]),np.multiply(alpha[l],xyz[l])),nu)
    nu_gamma = 1.0/(nu+gamma)
    rho = nu*gamma*nu_gamma
    pq = euclidean_norm(np.subtract(pab,qcd))
    t = rho*pq*pq
    if (t < eps):
       return res  
    #tdg_dt = 2.0*pow(t,0.5)*np.exp(-t) - 0.5*G
    tdg_dt = 0.5*np.exp(-t) - 0.5*G

    ## ab's
    
    pq = np.subtract(pab,qcd)

    ap = np.subtract(xyz[i],pab)
    da_dt_t = (1.0+2.0*(np.dot(pq,ap)/t*rho))/gamma - nu_gamma 
    res[i] += (tdg_dt*da_dt_t)


    ap = np.subtract(xyz[j],pab)
    da_dt_t = (1.0+2.0*(np.dot(pq,ap)/t*rho))/gamma - nu_gamma 
    res[j] += (tdg_dt*da_dt_t)


    ap = np.subtract(xyz[k],qcd)
    da_dt_t = (1.0-2.0*(np.dot(pq,ap)/t*rho))/nu - nu_gamma 
    res[k] += (tdg_dt*da_dt_t)


    ap = np.subtract(xyz[l],qcd)
    da_dt_t = (1.0-2.0*(np.dot(pq,ap)/t*rho))/nu - nu_gamma 
    res[l] += (tdg_dt*da_dt_t)

    return res

     
      

