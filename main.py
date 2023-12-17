# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:16:56 2023

@author: user
"""
import os
path=r'E:/Research/Code/Project1/'
os.chdir(path)

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
import pandas as pd
import os
import copy
from gravity_class import GRAVITY
#%%
if __name__ == '__main__':
    gravity=GRAVITY()
    out_intensity = np.load(path+'init_data/out_intensity.npy')   
    in_intensity = np.load(path+'init_data/in_intensity.npy')   
    beta = np.load(path+'init_data/beta.npy')
    
    X_all=np.load(path+'init_data/X_all.npy')
    SearchResult=np.load(path+'init_data/SearchResult.npy')
    d,p,N0,N1,Nx=out_intensity.shape[0],beta.shape[0],200,2000,X_all.shape[0]
    lamda,N_arl=0.1,10000
    
    #h=12
    #X=np.concatenate((np.ones((d,d,1),dtype=float),np.random.normal(0,0.2,(d,d,2))),2)
    k=10
    #k=np.random.randint(Nx)
    #h=SearchResult[k,0]
    h=101
    X=np.concatenate((np.ones((d,d,1),dtype=float),X_all[k,:,:,:]),2)
    for i in range(d):
        X[i,i,:]=0
    Expct=out_intensity.dot(in_intensity.T)*np.exp(np.dot(X,beta))[:,:,0]
    Expct=Expct-np.diag(np.diag(Expct))   ####self interacton
    XXT=np.zeros((d,d,p,p),dtype=float)
    for i in range(d):
        for j in range(d):
            XXT[i,j,:,:]=X[i:(i+1),j,:].T.dot(X[i:(i+1),j,:])   
    
    anomaly_type='beta0'
    SHIFT_beta0=[]#[-0.01,-0.02,-0.03,-0.04,-0.06,-0.08,-0.1,-0.12,0,0.01,0.02,0.03,0.04,0.06,0.08,0.1,0.12]
    for shift in SHIFT_beta0:
        beta_shifted=copy.deepcopy(beta)
        out_intensity_shifted=copy.deepcopy(out_intensity)
        in_intensity_shifted=copy.deepcopy(in_intensity)
        beta_shifted[0,0]+=shift
        Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
        Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
         
        def runlength_ALR(n_iter):
            Y=np.concatenate((gravity.network_gen(Expct,N0), gravity.network_gen(Expct_shifted,N1)),0)
            return gravity.chart_ALR(Y, X, XXT, N0, lamda, h)
            
        def ARL_ALR(N_arl):
            RL = Parallel(n_jobs=-1)(
                delayed(runlength_ALR)(n_iter) for n_iter in range(N_arl))
            return RL 
        
        RL=ARL_ALR(N_arl)

        result=pd.read_csv('result.csv')
        #result=pd.DataFrame(columns=['chart','k','N1','N0','lamda','h','anomaly_type','shift','N_arl','ARL','std'])
        result.loc[len(result)]=['ALR',k,N1,N0,lamda,h,anomaly_type,shift,N_arl,np.mean(RL),np.std(RL)/np.sqrt(len(RL))]
        result.to_csv('result.csv',index=False)
    

    anomaly_type='beta1'
    SHIFT_beta1=[0.25,0.3]#[-0.01,-0.02,-0.03,-0.05,-0.07,-0.1,-0.15,-0.2,-0.25,-0.3,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.25,0.3]
    for shift in SHIFT_beta1:
        beta_shifted=copy.deepcopy(beta)
        out_intensity_shifted=copy.deepcopy(out_intensity)
        in_intensity_shifted=copy.deepcopy(in_intensity)
        beta_shifted[1,0]+=shift
        Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
        Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
         
        def runlength_ALR(n_iter):
            Y=np.concatenate((gravity.network_gen(Expct,N0), gravity.network_gen(Expct_shifted,N1)),0)
            return gravity.chart_ALR(Y, X, XXT, N0, lamda, h)
            
        def ARL_ALR(N_arl):
            RL = Parallel(n_jobs=-1)(
                delayed(runlength_ALR)(n_iter) for n_iter in range(N_arl))
            return RL 
        
        RL=ARL_ALR(N_arl)

        result=pd.read_csv('result.csv')
        #result=pd.DataFrame(columns=['chart','k','N1','N0','lamda','h','anomaly_type','shift','N_arl','ARL','std'])
        result.loc[len(result)]=['ALR',k,N1,N0,lamda,h,anomaly_type,shift,N_arl,np.mean(RL),np.std(RL)/np.sqrt(len(RL))]
        result.to_csv('result.csv',index=False)        
    '''
    anomaly_type='a1'
    SHIFT_a0=[-0.02,-0.04,-0.06,-0.08,-0.1,-0.15,-0.2,-0.25,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.25]
    for shift in SHIFT_a0:
        beta_shifted=copy.deepcopy(beta)
        out_intensity_shifted=copy.deepcopy(out_intensity)
        in_intensity_shifted=copy.deepcopy(in_intensity)
        out_intensity_shifted[0,0]+=shift
        
        Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
        Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
         
        def runlength_ALR(n_iter):
            Y=np.concatenate((gravity.network_gen(Expct,N0), gravity.network_gen(Expct_shifted,N1)),0)
            return gravity.chart_ALR(Y, X, XXT, N0, lamda, h)
            
        def ARL_ALR(N_arl):
            RL = Parallel(n_jobs=-1)(
                delayed(runlength_ALR)(n_iter) for n_iter in range(N_arl))
            return RL 
        
        RL=ARL_ALR(N_arl)
        result=pd.read_csv('result.csv')
        #result=pd.DataFrame(columns=['chart','k','N1','N0','lamda','h','anomaly_type','shift','N_arl','ARL','std'])
        result.loc[len(result)]=['ALR',k,N1,N0,lamda,h,anomaly_type,shift,N_arl,np.mean(RL),np.std(RL)/np.sqrt(len(RL))]
        result.to_csv('result.csv',index=False)   
        
    anomaly_type='a1-a3'
    SHIFT_a03=[-0.02,-0.04,-0.06,-0.08,-0.1,-0.15,-0.2,-0.25,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.25]
    for shift in SHIFT_a03:
        beta_shifted=copy.deepcopy(beta)
        out_intensity_shifted=copy.deepcopy(out_intensity)
        in_intensity_shifted=copy.deepcopy(in_intensity)
        out_intensity_shifted[0:3,0]+=shift
        
        Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
        Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
         
        def runlength_ALR(n_iter):
            Y=np.concatenate((gravity.network_gen(Expct,N0), gravity.network_gen(Expct_shifted,N1)),0)
            return gravity.chart_ALR(Y, X, XXT, N0, lamda, h)
            
        def ARL_ALR(N_arl):
            RL = Parallel(n_jobs=-1)(
                delayed(runlength_ALR)(n_iter) for n_iter in range(N_arl))
            return RL 
        
        RL=ARL_ALR(N_arl)
        result=pd.read_csv('result.csv')
        #result=pd.DataFrame(columns=['chart','k','N1','N0','lamda','h','anomaly_type','shift','N_arl','ARL','std'])
        result.loc[len(result)]=['ALR',k,N1,N0,lamda,h,anomaly_type,shift,N_arl,np.mean(RL),np.std(RL)/np.sqrt(len(RL))]
        result.to_csv('result.csv',index=False) 
    '''

    '''
    anomaly_type='beta0'
    SHIFT_beta0=[-0.01,-0.02,-0.03,-0.04,-0.06,-0.08,-0.1,-0.12,0,0.01,0.02,0.03,0.04,0.06,0.08,0.1,0.12]
    for shift in SHIFT_beta0:
        beta_shifted=copy.deepcopy(beta)
        out_intensity_shifted=copy.deepcopy(out_intensity)
        in_intensity_shifted=copy.deepcopy(in_intensity)
        beta_shifted[0,0]+=shift
        Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
        Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
         
        def runlength_T2(n_iter):
            Y=np.concatenate((gravity.network_gen(Expct,N0), gravity.network_gen(Expct_shifted,N1)),0)
            return gravity.chart_T2(Y, X, XXT, N0, lamda, h)
            
        def ARL_T2(N_arl):
            RL = Parallel(n_jobs=-1)(
                delayed(runlength_T2)(n_iter) for n_iter in range(N_arl))
            return RL 
        
        RL=ARL_T2(N_arl)

        result=pd.read_csv('result.csv')
        #result=pd.DataFrame(columns=['chart','k','N1','N0','lamda','h','anomaly_type','shift','N_arl','ARL','std'])
        result.loc[len(result)]=['T2',k,N1,N0,lamda,h,anomaly_type,shift,N_arl,np.mean(RL),np.std(RL)/np.sqrt(len(RL))]
        result.to_csv('result.csv',index=False)
    

    anomaly_type='beta1'
    SHIFT_beta1=[-0.01,-0.02,-0.03,-0.05,-0.07,-0.1,-0.15,-0.2,-0.25,-0.3,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.25,0.3]
    for shift in SHIFT_beta1:
        beta_shifted=copy.deepcopy(beta)
        out_intensity_shifted=copy.deepcopy(out_intensity)
        in_intensity_shifted=copy.deepcopy(in_intensity)
        beta_shifted[1,0]+=shift
        Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
        Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
         
        def runlength_T2(n_iter):
            Y=np.concatenate((gravity.network_gen(Expct,N0), gravity.network_gen(Expct_shifted,N1)),0)
            return gravity.chart_T2(Y, X, XXT, N0, lamda, h)
            
        def ARL_T2(N_arl):
            RL = Parallel(n_jobs=-1)(
                delayed(runlength_T2)(n_iter) for n_iter in range(N_arl))
            return RL 
        
        RL=ARL_T2(N_arl)

        result=pd.read_csv('result.csv')
        #result=pd.DataFrame(columns=['chart','k','N1','N0','lamda','h','anomaly_type','shift','N_arl','ARL','std'])
        result.loc[len(result)]=['T2',k,N1,N0,lamda,h,anomaly_type,shift,N_arl,np.mean(RL),np.std(RL)/np.sqrt(len(RL))]
        result.to_csv('result.csv',index=False)        

    anomaly_type='a1'
    SHIFT_a0=[-0.02,-0.04,-0.06,-0.08,-0.1,-0.15,-0.2,-0.25,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.25]
    for shift in SHIFT_a0:
        beta_shifted=copy.deepcopy(beta)
        out_intensity_shifted=copy.deepcopy(out_intensity)
        in_intensity_shifted=copy.deepcopy(in_intensity)
        out_intensity_shifted[0,0]+=shift
        
        Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
        Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
         
        def runlength_T2(n_iter):
            Y=np.concatenate((gravity.network_gen(Expct,N0), gravity.network_gen(Expct_shifted,N1)),0)
            return gravity.chart_T2(Y, X, XXT, N0, lamda, h)
            
        def ARL_T2(N_arl):
            RL = Parallel(n_jobs=-1)(
                delayed(runlength_T2)(n_iter) for n_iter in range(N_arl))
            return RL 
        
        RL=ARL_T2(N_arl)
        result=pd.read_csv('result.csv')
        #result=pd.DataFrame(columns=['chart','k','N1','N0','lamda','h','anomaly_type','shift','N_arl','ARL','std'])
        result.loc[len(result)]=['T2',k,N1,N0,lamda,h,anomaly_type,shift,N_arl,np.mean(RL),np.std(RL)/np.sqrt(len(RL))]
        result.to_csv('result.csv',index=False)   
        
    anomaly_type='a1-a3'
    SHIFT_a03=[-0.02,-0.04,-0.06,-0.08,-0.1,-0.15,-0.2,-0.25,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.25]
    for shift in SHIFT_a03:
        beta_shifted=copy.deepcopy(beta)
        out_intensity_shifted=copy.deepcopy(out_intensity)
        in_intensity_shifted=copy.deepcopy(in_intensity)
        out_intensity_shifted[0:3,0]+=shift
        
        Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
        Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
         
        def runlength_T2(n_iter):
            Y=np.concatenate((gravity.network_gen(Expct,N0), gravity.network_gen(Expct_shifted,N1)),0)
            return gravity.chart_T2(Y, X, XXT, N0, lamda, h)
            
        def ARL_T2(N_arl):
            RL = Parallel(n_jobs=-1)(
                delayed(runlength_T2)(n_iter) for n_iter in range(N_arl))
            return RL 
        
        RL=ARL_T2(N_arl)
        result=pd.read_csv('result.csv')
        #result=pd.DataFrame(columns=['chart','k','N1','N0','lamda','h','anomaly_type','shift','N_arl','ARL','std'])
        result.loc[len(result)]=['T2',k,N1,N0,lamda,h,anomaly_type,shift,N_arl,np.mean(RL),np.std(RL)/np.sqrt(len(RL))]
        result.to_csv('result.csv',index=False) 
    '''   
                 

    '''
    anomaly_type='beta0'
    SHIFT_beta0=[-0.01,-0.02,-0.03,-0.04,-0.06,-0.08,-0.1,-0.12,0,0.01,0.02,0.03,0.04,0.06,0.08,0.1,0.12]
    for shift in SHIFT_beta0:
        beta_shifted=copy.deepcopy(beta)
        out_intensity_shifted=copy.deepcopy(out_intensity)
        in_intensity_shifted=copy.deepcopy(in_intensity)
        beta_shifted[0,0]+=shift
        Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
        Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
         
        def runlength_wlrt(n_iter):
            Y=np.concatenate((gravity.network_gen(Expct,N0), gravity.network_gen(Expct_shifted,N1)),0)
            return gravity.chart_WLRT(Y, X, XXT, N0, lamda, h)
            
        def ARL_wlrt(N_arl):
            RL = Parallel(n_jobs=-1)(
                delayed(runlength_wlrt)(n_iter) for n_iter in range(N_arl))
            return RL 
        
        RL=ARL_wlrt(N_arl)

        result=pd.read_csv('result.csv')
        #result=pd.DataFrame(columns=['chart','k','N1','N0','lamda','h','anomaly_type','shift','N_arl','ARL','std'])
        result.loc[len(result)]=['WLRT',k,N1,N0,lamda,h,anomaly_type,shift,N_arl,np.mean(RL),np.std(RL)/np.sqrt(len(RL))]
        result.to_csv('result.csv',index=False)
    

    anomaly_type='beta1'
    SHIFT_beta1=[-0.01,-0.02,-0.03,-0.05,-0.07,-0.1,-0.15,-0.2,-0.25,-0.3,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.2,0.25,0.3]
    for shift in SHIFT_beta1:
        beta_shifted=copy.deepcopy(beta)
        out_intensity_shifted=copy.deepcopy(out_intensity)
        in_intensity_shifted=copy.deepcopy(in_intensity)
        beta_shifted[1,0]+=shift
        Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
        Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
         
        def runlength_wlrt(n_iter):
            Y=np.concatenate((gravity.network_gen(Expct,N0), gravity.network_gen(Expct_shifted,N1)),0)
            return gravity.chart_WLRT(Y, X, XXT, N0, lamda, h)
            
        def ARL_wlrt(N_arl):
            RL = Parallel(n_jobs=-1)(
                delayed(runlength_wlrt)(n_iter) for n_iter in range(N_arl))
            return RL 
        
        RL=ARL_wlrt(N_arl)

        result=pd.read_csv('result.csv')
        #result=pd.DataFrame(columns=['chart','k','N1','N0','lamda','h','anomaly_type','shift','N_arl','ARL','std'])
        result.loc[len(result)]=['WLRT',k,N1,N0,lamda,h,anomaly_type,shift,N_arl,np.mean(RL),np.std(RL)/np.sqrt(len(RL))]
        result.to_csv('result.csv',index=False)        

    anomaly_type='a1'
    SHIFT_a0=[-0.02,-0.04,-0.06,-0.08,-0.1,-0.15,-0.2,-0.25,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.25]
    for shift in SHIFT_a0:
        beta_shifted=copy.deepcopy(beta)
        out_intensity_shifted=copy.deepcopy(out_intensity)
        in_intensity_shifted=copy.deepcopy(in_intensity)
        out_intensity_shifted[0,0]+=shift
        
        Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
        Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
         
        def runlength_wlrt(n_iter):
            Y=np.concatenate((gravity.network_gen(Expct,N0), gravity.network_gen(Expct_shifted,N1)),0)
            return gravity.chart_WLRT(Y, X, XXT, N0, lamda, h)
            
        def ARL_wlrt(N_arl):
            RL = Parallel(n_jobs=-1)(
                delayed(runlength_wlrt)(n_iter) for n_iter in range(N_arl))
            return RL 
        
        RL=ARL_wlrt(N_arl)
        result=pd.read_csv('result.csv')
        #result=pd.DataFrame(columns=['chart','k','N1','N0','lamda','h','anomaly_type','shift','N_arl','ARL','std'])
        result.loc[len(result)]=['WLRT',k,N1,N0,lamda,h,anomaly_type,shift,N_arl,np.mean(RL),np.std(RL)/np.sqrt(len(RL))]
        result.to_csv('result.csv',index=False)   
        
    anomaly_type='a1-a3'
    SHIFT_a03=[-0.02,-0.04,-0.06,-0.08,-0.1,-0.15,-0.2,-0.25,0.02,0.04,0.06,0.08,0.1,0.15,0.2,0.25]
    for shift in SHIFT_a03:
        beta_shifted=copy.deepcopy(beta)
        out_intensity_shifted=copy.deepcopy(out_intensity)
        in_intensity_shifted=copy.deepcopy(in_intensity)
        out_intensity_shifted[0:3,0]+=shift
        
        Expct_shifted=out_intensity_shifted.dot(in_intensity_shifted.T)*np.exp(np.dot(X,beta_shifted))[:,:,0]
        Expct_shifted=Expct_shifted-np.diag(np.diag(Expct_shifted))   ####self interacton
         
        def runlength_wlrt(n_iter):
            Y=np.concatenate((gravity.network_gen(Expct,N0), gravity.network_gen(Expct_shifted,N1)),0)
            return gravity.chart_WLRT(Y, X, XXT, N0, lamda, h)
            
        def ARL_wlrt(N_arl):
            RL = Parallel(n_jobs=-1)(
                delayed(runlength_wlrt)(n_iter) for n_iter in range(N_arl))
            return RL 
        
        RL=ARL_wlrt(N_arl)
        result=pd.read_csv('result.csv')
        #result=pd.DataFrame(columns=['chart','k','N1','N0','lamda','h','anomaly_type','shift','N_arl','ARL','std'])
        result.loc[len(result)]=['WLRT',k,N1,N0,lamda,h,anomaly_type,shift,N_arl,np.mean(RL),np.std(RL)/np.sqrt(len(RL))]
        result.to_csv('result.csv',index=False) 
    '''