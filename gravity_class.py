# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 10:58:26 2023

@author: user
"""
import numpy as np
from numpy import linalg
import copy

#%%
class GRAVITY:

    def network_gen(self,Expct,N):
        '''
        Parameters
        ----------
        Expct : d*d array
            The expectation matrix
        N : int
            number of sample

        Returns
        -------
        A : N*d*d array
            DESCRIPTION.

        '''
        from numpy.random import poisson
        d=Expct.shape[0]
        A=np.zeros((N,d,d),dtype=float)
        for i in range(d):
            for j in range(d):
                A[:,i,j]=poisson(Expct[i,j],N) 
        return A
    
    def newton(self,A,X,XXT,out_intensity_est,in_intensity_est,beta_est):
        '''
        Parameters
        ----------
        A : TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.
        XXT : TYPE
            DESCRIPTION.
        out_intensity_est : TYPE
            DESCRIPTION.
        in_intensity_est : TYPE
            DESCRIPTION.
        beta_est : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        d=A.shape[0]
        p=X.shape[2]
        update=np.ones((p,1),dtype=float)
      
        rate=5e-3
        newton_time=0
        while np.sqrt(linalg.norm(update,2)/linalg.norm(beta_est,2))>rate:
            if newton_time!=0:
                beta_est+=update
            expct=out_intensity_est.dot(in_intensity_est.T)*np.exp(np.dot(X,beta_est))[:,:,0]
            expct-=np.diag(np.diag(expct))
            
            Residual=np.reshape(A-expct,(d,d,1))
            gradiant=np.sum(X*Residual,(0,1)).reshape(-1,1)        #gradiant=np.zeros((p,1),dtype=float)
            
            expct=expct.reshape(d,d,1,1)
            hessian=-np.sum(expct*XXT,(0,1))

            update=-linalg.inv(hessian).dot(gradiant)
            newton_time+=1
        return beta_est+update

    def mle(self,A,X,XXT):# import d*d*p-dim array
        '''
        Parameters
        ----------
        A : TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.
        XXT : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
    
        d=A.shape[0]
        p=X.shape[2]
            
        D_out=np.sum(A,1).reshape(-1,1)
        D_in=np.sum(A,0).reshape(-1,1)
        
        in_intensity_init=np.zeros((d,1),dtype=float)
        out_intensity_init=np.zeros((d,1),dtype=float)
        beta_init=np.random.normal(0,1,(p,1))
        
        in_intensity_est=np.ones((d,1),dtype=float)
        out_intensity_est=np.ones((d,1),dtype=float)
        beta_est=0.5*np.ones((p,1),dtype=float)#np.random.normal(0,1,(p,1))#
       
        iter_time=0
        while np.sqrt(linalg.norm(np.concatenate((out_intensity_est-out_intensity_init,in_intensity_est-in_intensity_init,(beta_est-beta_init)),0),2)/linalg.norm(np.concatenate((out_intensity_init,in_intensity_init,(beta_init)),0),2))>1e-2:
            out_intensity_init=out_intensity_est
            in_intensity_init=in_intensity_est
            beta_init=beta_est
            ExpXbeta=np.exp(np.dot(X,beta_init))[:,:,0]
            ExpXbeta-=np.diag(np.diag(ExpXbeta))
            
            out_intensity_est=D_out/ExpXbeta.dot(in_intensity_init)
            in_intensity_est=D_in/ExpXbeta.T.dot(out_intensity_est)
            iter_time+=1
            if iter_time%3==0:    
                beta_est=self.newton(A,X,XXT,out_intensity_est, in_intensity_est, beta_init)
        
        if iter_time%3!=0:
            beta_est=self.newton(A,X,XXT,out_intensity_est, in_intensity_est, beta_init)

        beta_est[0,0]=beta_est[0,0]+np.log(np.sum(out_intensity_est)*np.sum(in_intensity_est)/d**2)
        out_intensity_est=out_intensity_est/np.sum(out_intensity_est)*d
        in_intensity_est=in_intensity_est/np.sum(in_intensity_est)*d 

        return out_intensity_est,in_intensity_est,beta_est

    def mle_seq(self,A,X,XXT,out_intensity_est,in_intensity_est,beta_est):# import d*d*p-dim array
        '''
        Parameters
        ----------
        A : TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.
        XXT : TYPE
            DESCRIPTION.
        out_intensity_est : TYPE
            DESCRIPTION.
        in_intensity_est : TYPE
            DESCRIPTION.
        beta_est : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        d=A.shape[0]
        p=X.shape[2]
            
        D_out=np.sum(A,1).reshape(-1,1)
        D_in=np.sum(A,0).reshape(-1,1)
        
        in_intensity_init=np.ones((d,1),dtype=float)
        out_intensity_init=np.ones((d,1),dtype=float)
        beta_init=0.5*np.ones((p,1),dtype=float)#np.random.normal(0,1,(p,1))#
       
        iter_time=0
        while np.sqrt(linalg.norm(np.concatenate((out_intensity_est-out_intensity_init,in_intensity_est-in_intensity_init,(beta_est-beta_init)),0),2)/linalg.norm(np.concatenate((out_intensity_init,in_intensity_init,(beta_init)),0),2))>1e-2:
            out_intensity_init=out_intensity_est
            in_intensity_init=in_intensity_est
            beta_init=beta_est
            ExpXbeta=np.exp(np.dot(X,beta_init))[:,:,0]
            ExpXbeta-=np.diag(np.diag(ExpXbeta))
            
            out_intensity_est=D_out/ExpXbeta.dot(in_intensity_init)
            in_intensity_est=D_in/ExpXbeta.T.dot(out_intensity_est)
            iter_time+=1
            if iter_time%3==0: 
                beta_est=self.newton(A,X,XXT,out_intensity_est, in_intensity_est, beta_init)
        
        if iter_time%3!=0: 
            beta_est=self.newton(A,X,XXT,out_intensity_est, in_intensity_est, beta_init)  

        beta_est[0,0]=beta_est[0,0]+np.log(np.sum(out_intensity_est)*np.sum(in_intensity_est)/d**2)
        out_intensity_est=out_intensity_est/np.sum(out_intensity_est)*d
        in_intensity_est=in_intensity_est/np.sum(in_intensity_est)*d 

        return out_intensity_est,in_intensity_est,beta_est

    def information(self,A,X,XXT,out_intensity,in_intensity,beta): #A: d*d  X:d*d*p
    
        p=beta.shape[0]
        index1,index2=list((out_intensity!=0).reshape(-1,)),list((in_intensity!=0).reshape(-1,))
        d1,d2=sum(index1),sum(index2)
     
        ExpXbeta=np.exp(np.dot(X,beta))[:,:,0]
        ExpXbeta-=np.diag(np.diag(ExpXbeta))
        ExpXbeta=ExpXbeta[index1,:]
        ExpXbeta=ExpXbeta[:,index2]  
       
        out_intensity=out_intensity[index1,:]
        in_intensity=in_intensity[index2,:]
        X=X[index1,:,:]
        X=X[:,index2,:]
        XXT=XXT[index1,:,:,:]
        XXT=XXT[:,index2,:,:]
        A=A[index1,:]
        A=A[:,index2]
        expct=out_intensity.dot(in_intensity.T)*ExpXbeta
        Residual=A-expct
        
        D_out=np.sum(A,1).reshape(-1,1)
        D_in=np.sum(A,0).reshape(-1,1)
        
        gradiant_out_intensity=D_out/out_intensity-ExpXbeta.dot(in_intensity)
        gradiant_in_intensity=D_in/in_intensity-ExpXbeta.T.dot(out_intensity)
        Residual=np.reshape(Residual,(d1,d2,1))
        gradiant_beta=np.sum(X*Residual,(0,1)).reshape(-1,1)        #gradiant=np.zeros((p,1),dtype=float)
      
        hessian=np.zeros((d1+d2+p,d1+d2+p),dtype=float) 
        hessian_out_intensity2=-D_out/(out_intensity**2)
        hessian_in_intensity2=-D_in/(in_intensity**2)   
        hessian_out_intensity_in_intensity=-ExpXbeta

        expctX=expct.reshape(d1,d2,1)*X#d1*d2*p
        hessian_out_intensity_beta=np.sum(expctX,1)/out_intensity##d1*p
        hessian_in_intensity_beta=np.sum(expctX,0)/in_intensity##d2*p
        hessian_beta2=-np.sum(expct.reshape(d1,d2,1,1)*XXT,(0,1))##p*p

        hessian[0:d1,0:d1]=np.diag(hessian_out_intensity2.reshape(-1,))
        
        hessian[d1:(d1+d2),0:d1]=hessian_out_intensity_in_intensity.T
        hessian[0:d1,d1:(d1+d2)]=hessian_out_intensity_in_intensity    
        hessian[d1:(d1+d2),d1:(d1+d2)]=np.diag(hessian_in_intensity2.reshape(-1,))
        
        hessian[(d1+d2):(d1+d2+p),0:d1]= hessian_out_intensity_beta.T 
        hessian[0:d1,(d1+d2):(d1+d2+p)]= hessian_out_intensity_beta   
        hessian[(d1+d2):(d1+d2+p),d1:(d1+d2)]= hessian_in_intensity_beta.T
        hessian[d1:(d1+d2),(d1+d2):(d1+d2+p)]= hessian_in_intensity_beta       
        hessian[(d1+d2):(d1+d2+p),(d1+d2):(d1+d2+p)]=hessian_beta2
        
        gradiant=np.concatenate((gradiant_out_intensity,gradiant_in_intensity,gradiant_beta),0)
        gradiant=np.delete(gradiant,[d1-1,d1+d2-1],0)
        hessian=np.delete(hessian,[d1-1,d1+d2-1],0)
        hessian=np.delete(hessian,[d1-1,d1+d2-1],1)
        
        return gradiant,-hessian
    
    def chart_WLRT(self,sample,X,XXT,N0,lamda,h):
        sample_weighted=np.mean(sample[:N0,:,:],0)
        sample=sample[N0:,:,:]
        N1=sample.shape[0]
        
        out_intensity_seq,in_intensity_seq,beta_seq=self.mle(sample_weighted,X,XXT)
        
        expct_ic_est=out_intensity_seq.dot(in_intensity_seq.T)*np.exp(np.dot(X,beta_seq))[:,:,0]
        expct_ic_est=expct_ic_est-np.diag(np.diag(expct_ic_est))   ####self interacton
        
        expct_ic_est_zero=copy.deepcopy(expct_ic_est)
        expct_ic_est_zero[expct_ic_est_zero==0]=1
    
        lrt=0
        r=0
        while (lrt<h and r<N1):
    
            sample_weighted=lamda*sample[r,:,:]+(1-lamda)*sample_weighted
            out_intensity_seq,in_intensity_seq,beta_seq=self.mle_seq(sample_weighted,X,XXT,out_intensity_seq,in_intensity_seq,beta_seq)
            
            expct_seq=out_intensity_seq.dot(in_intensity_seq.T)*np.exp(np.dot(X,beta_seq))[:,:,0]
            expct_seq=expct_seq-np.diag(np.diag(expct_seq))
            
            expct_seq_zero=copy.deepcopy(expct_seq)
            expct_seq_zero[expct_seq_zero==0]=1
            
            lrt=2*np.sum(np.log(expct_seq_zero)*sample_weighted-expct_seq+expct_ic_est-np.log(expct_ic_est_zero)*sample_weighted)
            r +=1
        return r
    
    def chart_T2(self,sample,X,XXT,N0,lamda,h):
        p=X.shape[2]
        sample_weighted=np.mean(sample[:N0,:,:],0)
        sample=sample[N0:,:,:]
        N1=sample.shape[0]
    
        out_intensity_ic,in_intensity_ic,beta_ic=self.mle(sample_weighted,X,XXT)
        out_intensity_seq,in_intensity_seq,beta_seq=copy.deepcopy(out_intensity_ic),copy.deepcopy(in_intensity_ic),copy.deepcopy(beta_ic)

        t2_square=0
        r=0
        
        while (t2_square<h and r<N1):
            out_intensity_seq,in_intensity_seq,beta_seq=self.mle_seq(sample[r,:,:],X,XXT,out_intensity_seq,in_intensity_seq,beta_seq)
            gradiant,cov_seq_inv=self.information(sample[r,:,:],X,XXT,out_intensity_seq,in_intensity_seq,beta_seq)

            index1,index2=list((out_intensity_seq!=0).reshape(-1,)),list((in_intensity_seq!=0).reshape(-1,))
            d1,d2=sum(index1),sum(index2)
            theta=np.concatenate((out_intensity_seq[index1,:]-out_intensity_ic[index1,:],in_intensity_seq[index2,:]-in_intensity_ic[index2,:],beta_seq-beta_ic),0)
            theta=np.delete(theta,[d1-1,d1+d2-1],0)

            DF=d1+d2+p-2
            t2_square=(theta.T.dot(cov_seq_inv).dot(theta)[0,0]-DF)/np.sqrt(2*DF)

            r+=1
        return r    
    
    def chart_ALR(self,sample,X,XXT,N0,lamda,h):
        
        p=X.shape[2]
        sample_weighted=np.mean(sample[:N0,:,:],0)
        sample=sample[N0:,:,:]
        N1=sample.shape[0]
        
        out_intensity_ic,in_intensity_ic,beta_ic=self.mle(sample_weighted,X,XXT)
        index1,index2=list((out_intensity_ic!=0).reshape(-1,)),list((in_intensity_ic!=0).reshape(-1,))
        DF=sum(index1)+sum(index2)+p-2
        
        W=[-1e10]
        r=0
        while (W[-1]<h and  r<N1):
            sample_weighted=sample[r,:,:]*lamda+(1-lamda)*sample_weighted
            gradiant,information_seq=self.information(sample_weighted,X,XXT,out_intensity_ic,in_intensity_ic,beta_ic)
            W.append((gradiant.T.dot(np.linalg.inv(information_seq)).dot(gradiant)[0,0]-DF)/np.sqrt(2*DF))
            r+=1
        return r

    
'''
k=np.random.randint(500)
h=T2_Search[k,0]
X=np.concatenate((np.ones((d,d,1),dtype=float),X_all[k,:,:,:]),2)
for i in range(d):
    X[i,i,:]=0
Expct=out_intensity.dot(in_intensity.T)*np.exp(np.dot(X,beta))[:,:,0]
Expct=Expct-np.diag(np.diag(Expct))   ####self interacton
XXT=np.zeros((d,d,p,p),dtype=float)
for i in range(d):
    for j in range(d):
        XXT[i,j,:,:]=X[i:(i+1),j,:].T.dot(X[i:(i+1),j,:])
#Expct_shifted=copy.deepcopy(Expct)
Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
'''
    