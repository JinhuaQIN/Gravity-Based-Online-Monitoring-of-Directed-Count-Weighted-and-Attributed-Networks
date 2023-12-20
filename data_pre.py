# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 23:45:25 2023

@author: user
"""
import os
path=r'E:/Research/Code/Project1/'
os.chdir(path)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import seaborn as sns
import matplotlib.pyplot as plt
import re
import time
import copy
import networkx as nx
from joblib import Parallel, delayed
from gravity_class import GRAVITY
#%%manufacture company
edge_list= pd.read_csv(path+'manufacture_company/communication.csv')
edge_list[['Sender','Recipient','EventDate']]=edge_list['Sender;Recipient;EventDate'].str.split(';', expand=True)
edge_list=edge_list[['Sender','Recipient','EventDate']]
edge_list[['Sender','Recipient']]=edge_list[['Sender','Recipient']].astype(int)

edge_list['datetime']= pd.to_datetime(edge_list['EventDate'])
start_time = pd.to_datetime('2010-01-01 00:00:00')
edge_list['days'] = (edge_list['datetime']-start_time)
edge_list['days']=edge_list['days'].dt.days

edge_list=edge_list[['Sender','Recipient','days','datetime']]



reportsto=pd.read_csv(path+'manufacture_company/reportsto.csv')
reportsto[['reports','to']]=reportsto['ID;ReportsToID'].str.split(';', expand=True)
reportsto=reportsto[['reports','to']] 
reportsto['index']=reportsto['to'].apply(len)
reportsto=reportsto[reportsto['index']<10]
reportsto=reportsto[['reports','to']]
reportsto[['reports','to']]=reportsto[['reports', 'to']].astype(int)
del  start_time
############################################
#%%
from_=reportsto['reports'].tolist()
to_=reportsto['to'].tolist()
X=np.zeros((167,167,1))
for i in range(len(reportsto)):
    X[from_[i]-1,to_[i]-1,0]=1
    #X[to_[i]-1,from_[i]-1,0]=1

Mat=np.zeros((272,167,167))###network series
from_node=edge_list['Sender'].tolist()
to_node=edge_list['Recipient'].tolist()
days=edge_list['days'].tolist()
for i in range(len(days)):
    Mat[days[i]-1,from_node[i]-1,to_node[i]-1]+=1

del  from_,to_,from_node, to_node,days,i
#drop the holidays
index=[True]*Mat.shape[0]
for i in range(Mat.shape[0]):
    if i%7==0 or i%7==1:
        index[i]=False
index[93]=False
index[121]=False
index[152]=False
index[153]=False
#index[27]=False
#index[66]=False

i=0
while i<Mat.shape[0]:
    lll=[]
    if not index[i]:
        while not index[i]:
            lll.append(i)
            i+=1
        lll.append(i)
        Mat[i,:,:]=np.sum(Mat[lll,:,:],0)
    i+=1

Mat=Mat[index,:,:]
del index,lll,i
################drop the silent nodes
#%%
#70 138  *     165*165*190     人次
Mat_=copy.deepcopy(Mat)
Mat_[Mat_>0]=1
drop_index=list(np.sum(Mat_,(0,1))+np.sum(Mat_,(0,2))>1000)
#rop_index=list(np.sum(Mat,(0,2))+np.sum(Mat,(1,2))>1000)
Mat=Mat[:,drop_index,:]
Mat=Mat[:,:,drop_index]
X=X[drop_index,:,:]
X=X[:,drop_index,:]
del drop_index,edge_list, reportsto


#%% parameter estimation
lamda,N0=0.1,160
d,p=Mat.shape[1],X.shape[2]+1
X=np.concatenate((np.ones((d,d,1),dtype=float),X),2)

for i in range(d):
    X[i,i,:]=0
XXT=np.zeros((d,d,p,p),dtype=float)
for i in range(d):
    for j in range(d):
        XXT[i,j,:,:]=X[i:(i+1),j,:].T.dot(X[i:(i+1),j,:])
        
gravity=GRAVITY()
out_intensity_ic,in_intensity_ic,beta_ic=gravity.mle(np.mean(Mat[:160,:,:],0),X,XXT)

#%% WLRT statistics
N0=160
sample_weighted=np.mean(Mat[:N0,:,:],0)
sample=Mat[N0:,:,:]

out_intensity_seq,in_intensity_seq,beta_seq=gravity.mle(sample_weighted,X,XXT)
expct_ic_est=out_intensity_seq.dot(in_intensity_seq.T)*np.exp(np.dot(X,beta_seq))[:,:,0]
expct_ic_est=expct_ic_est-np.diag(np.diag(expct_ic_est))   ####self interacton
expct_ic_est_=copy.deepcopy(expct_ic_est)
expct_ic_est_[expct_ic_est_==0]=1
#Ns=20
#expct_MAT=np.zeros((43,190-N0+1))
#expct_MAT[:,0:1]=expct_ic_est[:,Ns:(Ns+1)]

LRT=[]
for r in range(sample.shape[0]):

    sample_weighted=sample[r,:,:]*lamda+(1-lamda)*sample_weighted
    out_intensity_seq,in_intensity_seq,beta_seq=gravity.mle_seq(sample_weighted,X,XXT,out_intensity_seq,in_intensity_seq,beta_seq)
    
    expct_seq=out_intensity_seq.dot(in_intensity_seq.T)*np.exp(np.dot(X,beta_seq))[:,:,0]
    expct_seq=expct_seq-np.diag(np.diag(expct_seq))
      
    #expct_MAT[:,(r+1):(r+2)]=expct_seq[:,Ns:(Ns+1)]

    LRT.append(2*np.sum(expct_ic_est-expct_seq))
    expct_seq[expct_seq==0]=1
    LRT[-1]+=(2*np.sum(sample_weighted*(np.log(expct_seq)-np.log(expct_ic_est_))))

#%%  T2 statistics
N0=160
sample_weighted=np.mean(Mat[:N0,:,:],0)
sample=Mat[N0:,:,:]

out_intensity_ic,in_intensity_ic,beta_ic=gravity.mle(sample_weighted,X,XXT)
out_intensity_seq,in_intensity_seq,beta_seq=copy.deepcopy(out_intensity_ic),copy.deepcopy(in_intensity_ic),copy.deepcopy(beta_ic)
gradiant,cov_seq_inv=gravity.information(sample_weighted,X,XXT,out_intensity_seq,in_intensity_seq,beta_seq)
t2_square=[]

for r in range(sample.shape[0]):
    sample_weighted=sample[r,:,:]#*lamda+(1-lamda)*sample_weighted
    out_intensity_seq,in_intensity_seq,beta_seq=gravity.mle_seq(sample_weighted,X,XXT,out_intensity_seq,in_intensity_seq,beta_seq)
    #gradiant,cov_seq_inv=gravity.information(sample[r,:,:],X,XXT,out_intensity_seq,in_intensity_seq,beta_seq)

    #index1,index2=list((out_intensity_seq!=0).reshape(-1,)),list((in_intensity_seq!=0).reshape(-1,))
    #d1,d2=sum(index1),sum(index2)
    #theta=np.concatenate((out_intensity_seq[index1,:]-out_intensity_ic[index1,:],in_intensity_seq[index2,:]-in_intensity_ic[index2,:],beta_seq-beta_ic),0)
    #theta=np.delete(theta,[d1-1,d1+d2-1],0)
    #DF=d1+d2+p-2
    theta=np.concatenate((out_intensity_seq-out_intensity_ic,in_intensity_seq-in_intensity_ic,beta_seq-beta_ic),0)
    theta=np.delete(theta,[d-1,d*2-1],0)
    DF=d*2+p-2
    
    t2_square.append((theta.T.dot(cov_seq_inv).dot(theta)[0,0]-DF)/np.sqrt(2*DF))

#%% ALR statistics
N0=160
sample_weighted=np.mean(Mat[:N0,:,:],0)
sample=Mat[N0:,:,:]

out_intensity_ic,in_intensity_ic,beta_ic=gravity.mle(sample_weighted,X,XXT)
index1,index2=list((out_intensity_ic!=0).reshape(-1,)),list((in_intensity_ic!=0).reshape(-1,))
DF=sum(index1)+sum(index2)+p-2

ALR=[]
for r in range(sample.shape[0]):
    sample_weighted=sample[r,:,:]#*lamda+sample_weighted*(1-lamda)
    gradiant,information_seq=gravity.information(sample_weighted,X,XXT,out_intensity_ic,in_intensity_ic,beta_ic)
    stat=(gradiant.T.dot(np.linalg.inv(information_seq)).dot(gradiant)[0,0]-DF)/np.sqrt(2*DF)
    ALR.append(stat)

#%%
import seaborn as sns
index=list(np.linspace(1,30,30))
sns.set_style("whitegrid")
sns.set_palette("Set2")

%matplotlib inline

plt.figure(figsize=(16,16))
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.5)


ax1=plt.subplot(gs[0,:2])
ax1.plot(index,LRT,color='black',marker = "o",markersize=3)
ax1.axhline(y=45,color='gray', linestyle='--')
plt.title('WLR-based EWMA')
plt.xlabel('Index')
plt.xlim([0,30])

ax2=plt.subplot(gs[0,2:])
ax2.plot(index,t2_square,color='black',marker = "o",markersize=3)
ax2.axhline(y=30.5,color='gray', linestyle='--')
plt.title("$T^2$")
plt.xlabel('Index')
plt.xlim([0,30])

ax3=plt.subplot(gs[1,1:3])
ax3.plot(index, W,color='black',marker = "o",markersize=3)
ax3.axhline(y=30,color='gray', linestyle='--')
plt.title('ALR')
plt.xlabel('Index')
plt.xlim([0,30])
