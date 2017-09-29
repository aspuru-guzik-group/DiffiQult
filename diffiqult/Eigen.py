import numpy as np
import algopy 

''' This routines were taken from:
Golub G H and Van Loan C F (1996) Matrix Computations (3rd Edition) 
Johns Hopkins University Press, Baltimore
def householder_vec(x):
'''
def householder_vec(x):
   ''' Algorithm 5.1.1 '''
   # We are assumming the option UPLO ='U' in dystrd in Lapack
   sigma = np.dot(np.transpose(x[1:]),x[1:])
   if sigma == 0:
     beta = 0.0
     v = np.array([1.0])
   else:
     mu = np.sqrt(pow(x[0],2.0) + sigma)
     if x[0] <= 0.0:
        v = np.array([x[0] + mu])
     else:     
        v = np.array([-sigma/(x[0] - mu)])
   beta = 2.0*pow(v[0],2.0)/(sigma + pow(v[0],2.0))
   v = np.concatenate((v,x[1:]),axis=0)
   v = np.divide(v,v[0])
   return v,beta
   

def householder_mat(x):
    ''' Algorithm 8.3.1'''
    # We are going to store just the upper part of the matrix
    t = []
    n = len(x)
    a = x
    def addrow(index,diag,value):
        for j in range(0,index):
           t.append(0.0)
        t.append(diag)
        t.append(value)
        for j in range(index+2,n):
           t.append(0.0)

    for i in range(0,n-2):
        temp = a[1:,0]
        addrow(i,a[0,0],euclidean_norm(temp)) 
        v,beta = householder_vec(temp)
        p = beta*np.dot(a[1:,1:],v)
           
        w =  p- np.dot(np.divide(beta*np.dot(np.transpose(p),v),2.0),v)
        v = np.reshape(v,(len(v),1))
        w = np.reshape(w,(len(w),1))
        a = a[1:,1:]- np.dot(v,np.transpose(w)) - np.dot(w,np.transpose(v)) 
    addrow(n-2,a[0,0],a[0,1]) 
    addrow(n-1,a[1,1],a[1,1]) 
    del t[-1]
    a = np.reshape(t,(n,n))
    #a = np.add(a,a.T) #subtract(np.add(S,S.T),diag)
    return a


def w_repeated(w):
    tool = 0.0 #1e-12
    for i in range(len(w)):
        for j in range(i+1,len(w)):
            if abs(w[i] - w[j]) > tool:
               return False
    return True
               

def eigensolver(A):
    eigsys = algopy.eigh(A)
    #if w_repeated(eigsys[0]):
    #     print 'I have not implement it repeated eigenvalues'
    #     exit()
    return eigsys


