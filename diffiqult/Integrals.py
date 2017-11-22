import math
import Tools
from Tools import *
import scipy as sc
import sys
import numpy as np
import algopy
from algopy import UTPM, zeros

def overlapmatrix(alpha,coef,xyz,l,nbasis,contr_list,dtype):
    S = algopy.zeros((nbasis,nbasis),dtype=dtype)
    cont_i = 0
    for i,ci in enumerate(contr_list):
         cont_j = cont_i
         for j,cj in enumerate(contr_list[i:]):
            S[i,j+i] = S[j+i,i] = overlap_contracted(alpha[cont_i:cont_i+ci],coef[cont_i:cont_i+ci],xyz[i],l[i],
                                                     alpha[cont_j:cont_j+cj],coef[cont_j:cont_j+cj],xyz[j+i],l[j+i])
            cont_j += cj
         cont_i += ci
    return S

def overlap_contracted(alphas,coefa,xyza,la,betas,coefb,xyzb,lb):
    '''This version '''
    s = 0
    for i in range(len(alphas)):
        for j in range(len(betas)):
            s = s + overlap_primitive(alphas[i],coefa[i],xyza,la,
                                    betas[j],coefb[j],xyzb,lb)
    return s

def overlap_primitive(alpha,coefa,A,la,beta,coefb,B,lb):
    ''' This version does not take into account at all the angula momentum '''
    gamma = np.float64(1.0)/(alpha+beta)
    inti = np.float64(1.0)

    for x,l1 in enumerate(la):
       l2 = lb[x]
       tmp = 1.0
       if l1 >0: 
           tmp = tmp*(B[x]-A[x])*beta*gamma 
       if l2 >0: 
           tmp = tmp*(A[x]-B[x])*alpha*gamma
       if l1 + l2 == 2:
           tmp = tmp + 0.5*gamma#*gamma 
       inti = tmp*inti
    ab = -1.0*euclidean_norm2(np.subtract(A,B))
    exp = np.exp((alpha*beta*ab)*gamma)
    inti= inti*np.power((np.pi*gamma),1.5)
    return exp*inti*coefa*coefb


def nuclearmatrix(alpha,coef,xyz,l,nbasis,charge,atoms,numatoms,contr_list,dtype):
    V = algopy.zeros((nbasis,nbasis),dtype=dtype)
    cont_i = 0
    for i,ci in enumerate(list(contr_list)):
         cont_j = cont_i
         for j,cj in enumerate(contr_list[i:]):
            V[i,j+i] = V[j+i,i] = nuclear_contracted(alpha[cont_i:cont_i+ci],coef[cont_i:cont_i+ci],xyz[i],l[i],
                                                     alpha[cont_j:cont_j+cj],coef[cont_j:cont_j+cj],xyz[j+i],l[j+i],
                                                     atoms,charge,numatoms)
            cont_j += cj
         cont_i += ci
    return V

def nuclear_contracted(alphas,coefa,xyza,la,betas,coefb,xyzb,lb,atoms,charge,numatoms):
    '''This version '''
    v = 0
    for i in range(len(alphas)):
        for j in range(len(betas)):
            v = v + nuclear_primitive(alphas[i],coefa[i],xyza,la,
                                    betas[j],coefb[j],xyzb,lb,
                                    atoms,charge,numatoms)
    return v
    


def nuclear_primitive(alpha,coefa,A,la,beta,coefb,B,lb,C,charge,numatoms):
    ''' This version just work with s and p orbitals '''
    gamma = alpha+beta
    gammainv = 1.0/gamma
    ab = -1.0*euclidean_norm2(np.subtract(A,B))
    S00 = coefa*coefb*np.exp(ab*alpha*beta*gammainv)*np.power((np.pi*gammainv),1.5)
    P = np.multiply(np.add(np.multiply(alpha,A),np.multiply(beta,B)),gammainv)
    n = max(la)
    m = max(lb)
    half_pi = np.sqrt(np.pi)*2
    if n+m == 0:
        nuclear = 0.0
        for i in range(numatoms):
            pc = gamma*euclidean_norm2(np.subtract(P,C[i]))
            nuclear = nuclear -  charge[i]*incompletegammaf(pc,0.0)
            #nuclear = nuclear -  charge[i]*algopy.erf(pc)
        return nuclear*S00*2.0*np.power(gamma/np.pi,0.5)
    if n+m == 2:
       i = la.index(n)
       j = lb.index(m)
       Si0 = (B[i]-A[i])*beta*gammainv
       S0j = (A[j]-B[j])*alpha*gammainv
       Sij = S0j*Si0
       if i == j:
          Sij = Sij + 0.5*gammainv
       nuclear = 0.0
       for k in range(numatoms):
          tmp = 0.0
          pc = gamma*euclidean_norm2(np.subtract(P,C[k]))
          L00 = incompletegammaf(pc,0.0)
          L0j = (C[k,j]-P[j])
          Li0 = (C[k,i]-P[i])
          Lij = L0j*Li0*incompletegammaf(pc,2.0)
          L0j = L0j*incompletegammaf(pc,1.0)
          Li0 = Li0*incompletegammaf(pc,1.0)
          if i == j:
             Lij = Lij - 0.5*gammainv*incompletegammaf(pc,1.0) 
          nuclear= nuclear - charge[k]*(Sij*L00+Si0*L0j+S0j*Li0+Lij)
       return nuclear*S00*2.0*np.power(gamma/np.pi,0.5)
    if n == 1:
       i = la.index(n)
       Si0 = (B[i]-A[i])*beta*gammainv 
       nuclear = 0.0
       for k in range(0,numatoms):
          pc = gamma*euclidean_norm2(np.subtract(P,C[k]))
          L00 = incompletegammaf(pc,0.0)
          Li0 = np.subtract(C[k,i],P[i])*incompletegammaf(pc,1.0)
          nuclear = nuclear - charge[k]*(Li0 + L00*Si0)
       return nuclear*2.0*np.sqrt(gamma/np.pi)*S00
    if m == 1:
       j = lb.index(m)
       S0j = (A[j]-B[j])*alpha*gammainv
       nuclear = 0.0
       for k in range(numatoms):
          tmp = 0.0
          pc = gamma*euclidean_norm2(np.subtract(P,C[k]))
          L00 = incompletegammaf(pc,0.0)
          L0j = (C[k,j]-P[j])*incompletegammaf(pc,1.0)
          nuclear =nuclear -  charge[k]*(S0j*L00+L0j)
       return nuclear*S00*2.0*pow(gamma/np.pi,0.5)

def kineticmatrix(alpha,coef,xyz,l,nbasis,contr_list,dtype):
    T = algopy.zeros((nbasis,nbasis),dtype=dtype)
    cont_i = 0
    for i,ci in enumerate(contr_list):
         cont_j = cont_i
         for j,cj in enumerate(contr_list[i:]):
            T[i,j+i] = T[j+i,i] = kinetic_contracted(alpha[cont_i:cont_i+ci],coef[cont_i:cont_i+ci],xyz[i],l[i],
                                                     alpha[cont_j:cont_j+cj],coef[cont_j:cont_j+cj],xyz[j+i],l[j+i])
            cont_j += cj
         cont_i += ci
    return T

def kinetic_contracted(alphas,coefa,xyza,la,betas,coefb,xyzb,lb):
    '''This version '''
    t = 0.0
    for i in range(len(alphas)):
        for j in range(len(betas)):
            t = t + kinetic_primitive(alphas[i],coefa[i],xyza,la,
                                       betas[j],coefb[j],xyzb,lb)
    return t

def kinetic_primitive(alpha,coefa,A,la,beta,coefb,B,lb):
    ''' This version just work with s and p orbitals '''
    gamma = 1.0/(alpha+beta)
    eta = alpha*beta*gamma
    ab = -1.0*euclidean_norm2(np.subtract(A,B))
    S00 = coefa*coefb*np.exp(ab*eta)*np.power((np.pi*gamma),1.5)
    K00 = 3.0*eta + 2.0*eta*eta*ab
    n = max(la)
    m = max(lb)
    if n+m == 0:
       return S00*K00
    if n+m == 2:
       i = la.index(n)
       j = lb.index(m)
       Ki0 = 2.0*eta*gamma*beta*(B[i]-A[i])
       K0j = 2.0*eta*gamma*alpha*(A[j]-B[j])
       Si0 = (B[i]-A[i])*beta*gamma 
       S0j = (A[j]-B[j])*alpha*gamma
       Sij = S0j*Si0
       tot = S0j*Ki0 + Si0*K0j 
       if i == j:
          Kij = eta*gamma
          Sij = Sij + 0.5*gamma
          return S00*(tot + Sij*K00 + Kij)
       return S00*(tot + Sij*K00)
    if n == 1:
       i = la.index(n)
       Ki0 = 2.0*eta*gamma*beta*(B[i]-A[i])
       Si0 = (B[i]-A[i])*beta*gamma 
       return (Si0*K00 + Ki0)*S00
    if m == 1:
       j = lb.index(m)
       K0j = 2.0*eta*gamma*alpha*(A[j]-B[j])
       S0j = (A[j]-B[j])*alpha*gamma
       return (S0j*K00 + K0j)*S00

def erivector(alpha,coef,xyz,l,nbasis,contr_list,dtype):
    '''This function returns the eris in form a vector, 
    to get an element of the eris tensor, use eri_index, 
    included in Tools'''
    ### NOTE: YOU CAN REPLACE A VALUE OF THE ARRAY!!!
    vec_size = nbasis*(nbasis**3 + 2*nbasis**2 + 3*nbasis + 2)/8 ## Vector size
    Eri_vec = algopy.zeros((vec_size,),dtype=dtype)
    len_vec = nbasis*(nbasis + 1)/2 
    for x in range(len_vec):
        i,j = vec_tomatrix(x,nbasis)
        contr_i_i = sum(contr_list[0:i])
        contr_i_f = sum(contr_list[0:i+1])
        contr_j_i = sum(contr_list[0:j])
        contr_j_f = sum(contr_list[0:j+1])
        for y in range(x,len_vec) :
             k,m = vec_tomatrix(y,nbasis)
             contr_m_i = sum(contr_list[0:m])
             contr_m_f = sum(contr_list[0:m+1])
             contr_k_i = sum(contr_list[0:k])
             contr_k_f = sum(contr_list[0:k+1])
             index = matrix_tovector(x,y,len_vec)
             Eri_vec[index] = eri_contracted(
                                         alpha[contr_i_i:contr_i_f],coef[contr_i_i:contr_i_f],xyz[i],l[i],
                                         alpha[contr_j_i:contr_j_f],coef[contr_j_i:contr_j_f],xyz[j],l[j],
                                         alpha[contr_k_i:contr_k_f],coef[contr_k_i:contr_k_f],xyz[k],l[k],
                                         alpha[contr_m_i:contr_m_f],coef[contr_m_i:contr_m_f],xyz[m],l[m])
    return Eri_vec

def eri_contracted(alphas,coefa,A,la,betas,coefb,B,lb,
                   kappas,coefk,C,lk,nus,coefn,D,ln):
    '''This version '''
    eris = 0.0
    for i in range(len(alphas)):
        for j in range(len(betas)):
            for k in range(len(kappas)):
                 for n in range(len(nus)):
                     eris = eris + eris_primitive(alphas[i],coefa[i],A,la,
                                             betas[j],coefb[j],B,lb,
                                            kappas[k],coefk[k],C,lk,
                                               nus[n],coefn[n],D,ln)
    return eris



def eris_primitive(a,coefa,A,la,b,coefb,B,lb,c,coefc,C,lc,d,coefd,D,ld):
    ''' This version does not take into account at all the angula momentum '''
    eps = 1e-5
    gamma = a+b
    gammainv = 1.0/gamma
    pab = np.multiply(np.add(np.multiply(a,A),np.multiply(b,B)),gammainv)
    fab = coefa*coefb
    ab = euclidean_norm2(np.subtract(A,B))*gammainv
    kab = fab*np.exp(-1.0*a*b*ab)*gammainv

    nu = c+d
    nuinv = 1.0/nu
    qcd = np.multiply(np.add(np.multiply(c,C),np.multiply(d,D)),nuinv)
    fcd = coefc*coefd
    cd = euclidean_norm2(np.subtract(C,D))*nuinv
    kcd = fcd*np.exp(-1.0*c*d*cd)*nuinv

    nugammainv = 1.0/(nu+gamma)
    rho = nu*gamma*nugammainv
    t = rho*euclidean_norm2(np.subtract(pab,qcd))
    #print t

    na = max(la)
    nb = max(lb)
    nc = max(lc)
    nd = max(ld)
 
    def F_m(terms):
       F = []
       for i in range(terms+1):
          F.append(incompletegammaf(t,i))
       return F
    
    Fs = F_m(na+nb+nc+nd)
    #print Fs

    if (len(Fs) == 1):
       return kcd*kab*2.0*pow(np.pi,2.5)*Fs[0]*np.sqrt(nugammainv)

    prefactor =  kcd*kab*2.0*pow(np.pi,2.5)*np.sqrt(nugammainv)
    wpq= np.multiply(np.add(np.multiply(gamma,pab),np.multiply(nu,qcd)),nugammainv)

    def psss(i,P,A_xyz,m):
       #print 'I am calculating psss'
       res = []
       PA =P[i]-A_xyz[i]
       WP = wpq[i]-P[i]
       for n in range(m+1):
           res.append(PA*Fs[0+n]+WP*Fs[1+n])
       #print 'I entered psss',res,PA,WP
       return res

    def ppss(i,j,P,A_xyz,B_xyz,ginv,m):
       #print 'I am calculating ppss'
       psss_aux = psss(i,P,A_xyz,m+1)
       res = []
       PB =P[j]-B_xyz[j]
       WP = wpq[j]-P[j]
       for n in range(m+1):
          tmp = PB*psss_aux[n]+WP*psss_aux[n+1]
          if (i==j):
              tmp = tmp + 0.5*ginv*(Fs[0+n]- rho*ginv*Fs[n+1])
          res.append(tmp)
       #print 'I entered ppss',res
       return res
        
    def psps(i,k,P,Q,A_xyz,C_xyz,nginv,m):
       #print 'I am calculating psps'
       psss_aux = psss(i,P,A_xyz,m+1)
       res = []
       QC =Q[k]-C_xyz[k]
       WQ = wpq[k]-Q[k]
       for n in range(m+1):
          tmp = QC*psss_aux[n]+WQ*psss_aux[n+1]
          if (i==k):
              tmp = tmp + 0.5*nginv*(Fs[1+n])
          res.append(tmp)
       #print 'I entered psps',res
       return res
    
    def ppps(i,j,k,P,Q,A_xyz,B_xyz,C_xyz,ginv,m): 
       ppss_aux = ppss(i,j,P,A_xyz,B_xyz,ginv,m+1)
       QC =Q[k]-C_xyz[k]
       WQ = wpq[k]-Q[k]
       res = []
       if (i==k):
          spss_aux = psss(j,P,B_xyz,m+1)
       if (j==k):
          psss_aux = psss(i,P,A_xyz,m+1)
       for n in range(m+1):
           tmp = QC*ppss_aux[n]+WQ*ppss_aux[n+1]
           if (i==k):
               tmp = tmp + 0.5*nugammainv*spss_aux[n+1]
           if (j==k):
               tmp = tmp + 0.5*nugammainv*psss_aux[n+1]
           res.append(tmp)
       return res



    def pppp(m):
       #print 'I am calculating pppp'
       i = la.index(na)
       j = lb.index(nb)
       k = lc.index(nc)
       l = ld.index(nd)
       #ppps_aux = ppps(i,j,k,np.copy(pab),np.copy(qcd),np.copy(A),np.copy(B),np.copy(C),gammainv,1) 
       ppps_aux = ppps(i,j,k,pab,qcd,A,B,C,gammainv,1) 
       QD =qcd[l]-D[l]
       WQ =wpq[l]-qcd[l] 
       #print 'angular',i,j,k,l
       tmp = QD*ppps_aux[0]+WQ*ppps_aux[1]
       #print 'tmp in pppp',tmp
       if (i==l):
          #print '1 i==l'
          spps_aux = psps(j,k,pab,qcd,B,C,nugammainv,1) 
          tmp = tmp + 0.5*nugammainv*spps_aux[1]
       if (j==l):
          #print '2 j==i'
          #print A,C,i,k
          #print nugammainv,tmp
                 #def psps(i,k,P,Q,A_xyz,C_xyz,nginv,m):
          psps_aux = psps(i,k,pab,qcd,A,C,nugammainv,1) 
          tmp = tmp + 0.5*nugammainv*psps_aux[1]
       if (k==l):
          #print '3 k==l'
          ppss_aux = ppss(i,j,pab,A,B,gammainv,1) 
          tmp = tmp + 0.5*nuinv*(ppss_aux[0]-nuinv*rho*ppss_aux[1])
       #print 'I entered pppp',tmp
       return tmp 
        
    if (len(Fs) == 2):
      if nd == 1:
         return prefactor*psss(ld.index(nd),qcd,D,0)[0]
      if nc == 1:
         return prefactor*psss(lc.index(nc),qcd,C,0)[0]
      if nb == 1:
         return prefactor*psss(lb.index(nb),pab,B,0)[0]
      if na == 1:
         return prefactor*psss(la.index(na),pab,A,0)[0]

    if (len(Fs) == 3):
        if (nd+nc == 2):
            return prefactor*ppss(lc.index(nc),ld.index(nd),qcd,C,D,nuinv,0)[0]
        if (na+nb == 2):
            return prefactor*ppss(la.index(na),lb.index(nb),pab,A,B,gammainv,0)[0]
        if (nb+nd == 2):
            return prefactor*psps(lb.index(nb),ld.index(nd),pab,qcd,B,D,nugammainv,0)[0]
        if (nb+nc == 2):
            return prefactor*psps(lb.index(nb),lc.index(nc),pab,qcd,B,C,nugammainv,0)[0]
        if (na+nc == 2):
            return prefactor*psps(la.index(na),lc.index(nc),pab,qcd,A,C,nugammainv,0)[0]
        return prefactor*psps(la.index(na),ld.index(nd),pab,qcd,A,D,nugammainv,0)[0]
    
    if (len(Fs) == 5):
       return pppp(0)*prefactor
    
    if (nd+nc == 2):
       if (nb == 1):
          return prefactor*ppps(lc.index(nc),ld.index(nd),lb.index(nb),qcd,pab,C,D,B,nuinv,0)[0] 
       return prefactor*ppps(lc.index(nc),ld.index(nd),la.index(na),qcd,pab,C,D,A,nuinv,0)[0] 
    elif (na+nb == 2):
       if (nc == 1):
           return prefactor*ppps(la.index(na),lb.index(nb),lc.index(nc),pab,qcd,A,B,C,gammainv,0)[0] 
       else:
           return prefactor*ppps(la.index(na),lb.index(nb),ld.index(nd),pab,qcd,A,B,D,gammainv,0)[0] 
      
    return 1.0
    


def erissmatrix3(alpha,xyz,nbasis,S):
    G = np.multiply(2.0,erissg(alpha,xyz,nbasis))
    A = erisa(alpha,nbasis,S)
    return np.multiply(A,G)

def erisanew(alpha,nbasis,S):
    Sab = np.multiply(np.eye(nbasis),alpha)
    #print Sab
    exit()
def erisa(alpha,nbasis,S):
    A = []
    for i in range(nbasis):
       for j in range(nbasis):
          gamma = alpha[i]+alpha[j]
          Sab = S[i][j]
          for k in range(nbasis):
             for l in range(nbasis):
                 Scd = S[k][l]
                 nu = alpha[k]+alpha[l]
                 A.append(Sab*Scd*np.sqrt(nu*gamma/(np.pi*(nu+gamma))))
    A = np.reshape(np.array(A),(nbasis,nbasis,nbasis,nbasis))
    return A

def erissg(alpha,xyz,nbasis):
    G = []
    for i in range(nbasis):
       for j in range(nbasis):
          for k in range(nbasis):
             for l in range(nbasis):
                 G.append(gelem(alpha[i],xyz[i],
                                alpha[j],xyz[j],
                                alpha[k],xyz[k],
                                alpha[l],xyz[l]))
    G = np.reshape(np.array(G),(nbasis,nbasis,nbasis,nbasis))
    return G


def gelem(a,A,b,B,c,C,d,D):
    ''' This version does not take into account at all the angula momentum '''
    eps = 1e-5
    gamma = a+b
    pab = np.divide(np.add(np.multiply(a,A),np.multiply(b,B)),gamma)

    nu = c+d
    qcd = np.divide(np.add(np.multiply(c,C),np.multiply(d,D)),nu)

    rho = nu*gamma/(nu+gamma)
    t = rho*euclidean_norm2(np.subtract(pab,qcd))

    if t  > eps :
       tmp =  np.float64(0.5)*np.sqrt(np.pi/t)*sc.special.erf(np.sqrt(t))
    else:
       tmp = 1.0
    
    return tmp



def normalization_primitive(alpha,l,l_large):
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


def normalization(alpha,c,l,list_contr,dtype=np.float64(1.0)):
    contr = 0
    coef = algopy.zeros(len(alpha),dtype=dtype)
    
    for ib, ci in enumerate(list_contr):
       div = 1.0
       l_large = 0
       for m in l[ib]:
          if (m>0):
            l_large += 1
            for k in range(1,2*m,2):
              div = div*k
       div = div/pow(2,l_large)
       div = div*pow(np.pi,1.5)
       for i in range(ci):
           coef[contr+i] = normalization_primitive(alpha[contr+i],l[ib],l_large)*c[contr+i]
       #print l_large
       #print 'div',div
       tmp = 0.0
       for i in range(ci):
          for j in range(ci):
               tmp = tmp+ coef[contr+j]*coef[contr+i]/np.power(alpha[contr+i]+alpha[contr+j],l_large+1.5)
       tmp = algopy.sqrt(tmp*div) 
       for i in range(contr,contr+ci):
            coef[i] = coef[i]/tmp
       contr = contr + ci
    return coef


def AO_to_MO(integral,MO_coef):
    '''This function returns the MO-Eris in form a vector, 
    to get an element of the eris tensor, use eri_index, 
    included in Tools'''
    ### NOTE: YOU CAN REPLACE A VALUE OF THE ARRAY!!!
    vec_size = nbasis*(nbasis**3 + 2*nbasis**2 + 3*nbasis + 2)/8 ## Vector size
    Eri_vec = algopy.zeros((vec_size,),dtype=dtype)
    len_vec = nbasis*(nbasis + 1)/2
    for x in range(len_vec):
        i,j = vec_tomatrix(x,nbasis)
        contr_i_i = sum(contr_list[0:i])
        contr_i_f = sum(contr_list[0:i+1])
        contr_j_i = sum(contr_list[0:j])
        contr_j_f = sum(contr_list[0:j+1])
        for y in range(x,len_vec) :
             k,m = vec_tomatrix(y,nbasis)
             contr_m_i = sum(contr_list[0:m])
             contr_m_f = sum(contr_list[0:m+1])
             contr_k_i = sum(contr_list[0:k])
             contr_k_f = sum(contr_list[0:k+1])
             index = matrix_tovector(x,y,len_vec)
             Eri_vec[index] = eri_contracted(
                                         alpha[contr_i_i:contr_i_f],coef[contr_i_i:contr_i_f],xyz[i],l[i],
                                         alpha[contr_j_i:contr_j_f],coef[contr_j_i:contr_j_f],xyz[j],l[j],
                                         alpha[contr_k_i:contr_k_f],coef[contr_k_i:contr_k_f],xyz[k],l[k],
                                         alpha[contr_m_i:contr_m_f],coef[contr_m_i:contr_m_f],xyz[m],l[m])
    return Eri_vec
