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

Mat=np.zeros((167,167,272))
from_node=edge_list['Sender'].tolist()
to_node=edge_list['Recipient'].tolist()
days=edge_list['days'].tolist()
for i in range(len(days)):
    Mat[from_node[i]-1,to_node[i]-1,days[i]-1]+=1

del  from_,to_,from_node, to_node,days,i
#drop the holidays
index=[True]*Mat.shape[2]
for i in range(Mat.shape[2]):
    if i%7==0 or i%7==1:
        index[i]=False
index[93]=False
index[121]=False
index[152]=False
index[153]=False
#index[27]=False
#index[66]=False

i=0
while i<Mat.shape[2]:
    lll=[]
    if not index[i]:
        while not index[i]:
            lll.append(i)
            i+=1
        lll.append(i)
        Mat[:,:,i]=np.sum(Mat[:,:,lll],2)
    i+=1

Mat=Mat[:,:,index]
del index,lll,i
################drop the silent nodes
#%%
#70 138  *     165*165*190     人次
Mat_=copy.deepcopy(Mat)
Mat_[Mat_>0]=1
drop_index=list(np.sum(Mat_,(0,2))+np.sum(Mat_,(1,2))>1000)
#rop_index=list(np.sum(Mat,(0,2))+np.sum(Mat,(1,2))>1000)
Mat=Mat[drop_index,:,:]
Mat=Mat[:,drop_index,:]
X=X[drop_index,:,:]
X=X[:,drop_index,:]
del drop_index,edge_list, reportsto