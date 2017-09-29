import numpy
import autograd
from autograd.util import *
import autograd.numpy.random as npr
from autograd import grad
import Eigen
from Eigen import householder_vec, householder_mat

def fake_vec(n,scale=1.0):
    return scale*np.random.random(size=n)

def fake_matrix(n,scale=1.0):
    mat = np.zeros((n,n),dtype='float64')
    for i in range(n):
       for j in range(i,n):
           mat[i,j]=mat[j,i]=np.random.random()
    return scale*mat
   
def fake_grad(x):
   x = fake_matrix(x)
   w, v = np.linalg.eigh(x)
   print 'eigen',w
   return to_scalar(w) + to_scalar(v)

def test_householder_mat():
   tool = 2e-26
   def check_mat_tdiag(a):
       n = len(a)
       for i in range(n-1,2,-1):
         for j in range(i+2,n-2):
             if a[i,j] != 0:
                print i,j,n
                print 'Error in householder_vec'
                print a[i,j]
                exit()
             if a[j,i] != 0:
                print 'Error in householder_vec'
                print a[j,i]
                exit()
   ## Example 8.3.1
   x = np.array([[1.,3.,4.], [3.,2.,8.], [4.,8.,3.]])
   a = householder_mat(x)
   
   for i in range(3,800,101):
      x = fake_matrix(i)
      a = householder_mat(x)
      check_mat_tdiag(a)
   return 


def test_householder_vec():
   tool = 2e-26
   def check_P(v,b):
      n = len(v)
      v = np.reshape(v,(n,1))
      p_matrix = np.subtract(np.eye(n,dtype='float64'),b*np.dot(v,np.transpose(v)))
      new_x = np.dot(p_matrix,x)
      if np.dot(np.transpose(new_x[1:]),new_x[1:]) > tool:
         print 'Error in householder_vec'
         print np.dot(np.transpose(new_x[1:]),new_x[1:]) 
         exit()
      if new_x[0] == 0.0:
         print 'Error in householder_vec'
         exit()
      print new_x,np.dot(np.transpose(new_x[1:]),new_x[1:]) 
  
   #eg. page 209
   x = np.array([3.0,1.0,5.0,1.0]) 
   v,b = householder_vec(x)
   check_P(v,b)

   for i in range(3,800,101):
      x = fake_vec(i)
      v,b = householder_vec(x)
      check_P(v,b)
   return 
      
if __name__ == '__main__':
   
   #test_householder_vec()
   test_householder_mat()
   exit()
   x = np.array([3.5,3.5,3.5])
   print fake_grad(x)
   grad_f = grad(fake_grad)
   res = grad_f(x)
   print res
