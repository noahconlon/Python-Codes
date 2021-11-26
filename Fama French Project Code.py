#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 16:19:42 2021

@author: noahconlon
"""

import pandas as pd   
import numpy as np   
import scipy as scp  
import matplotlib.pyplot as plt  
import os                as os   

from datetime import date as dd  # for dates
from scipy import stats    # stats module
from scipy import optimize # op
import statsmodels.api as sm

df = pd.read_csv('/Users/noahconlon/Downloads/Project Data.csv')

df = df.drop(['vwretd', 'RET'], axis = 1)


df.columns = ['permno', 'date', 'name', 'Mkt-Rf', 'SMB', 'HML', 'RMW', 'CMA', 'Rf', 'Return']
df['Ret-Rf'] = df['Return']-df['Rf']
permno = list(dict.fromkeys(df['permno']))

betas = pd.DataFrame(permno, columns = ['permno']) 
vals = ['ION', 'QCOM','JACK', 'RES', 'NEURO', 'SEMPR', 'IlMNA', 'ACADIA', 'BIO', 'DCOM']
betas['Name'] = vals
betas['beta'] = np.nan
betas['FBeta'] = np.nan
betas['FR_squared'] = np.nan
betas['pvalue'] = np.nan
betas['R_squared'] = np.nan
betas['TR_squared'] = np.nan
print(df.shape)
print(df['Mkt-Rf'].shape)




for i in permno:
    subset = df[df['permno']==i].dropna()
    x0 = np.array(subset['Mkt-Rf'])
    x1 = sm.add_constant(np.array(subset[['Mkt-Rf','SMB','HML','RMW','CMA']]))
    x2 = sm.add_constant(np.array(subset[['Mkt-Rf','SMB','HML']]))
    y = np.array(subset['Ret-Rf'])
    

    capm = sm.OLS(y, x0).fit()
    mod = sm.OLS(y,x1).fit()
    threemod = sm.OLS(y,x2).fit()

    betas.loc[betas['permno'] == i, 'beta'] = capm.params[0]
    betas.loc[betas['permno'] == i, 'FBeta'] = mod.params[1]
    betas.loc[betas['permno']==i, 'R_squared'] = capm.rsquared
    betas.loc[betas['permno']==i, 'FR_squared'] = mod.rsquared
    betas.loc[betas['permno']==i, 'TR_squared'] = threemod.rsquared
    print(mod.summary())
    
fig = plt.figure()
plt.plot(betas['Name'], betas['FR_squared'], label = 'Five Factors')
plt.plot(betas['Name'], betas['R_squared'], label = 'CAPM')
plt.title('Five Factor vs. CAPM')
plt.xlabel('Company Name')
plt.ylabel('R-squared')
plt.legend(loc = 'upper left')
plt.xticks(rotation = 45)

fig = plt.figure()
plt.plot(betas['Name'], betas['FR_squared'], label = 'Five Factors')
plt.plot(betas['Name'], betas['TR_squared'], label = 'Three Factors')
plt.title('Five Factor vs. Three Factor')
plt.xlabel('Company Name')
plt.ylabel('R-squared')
plt.legend(loc = 'upper left')
plt.xticks(rotation = 45)

fig = plt.figure()
plt.plot(betas['Name'], betas['TR_squared'], label = 'Three Factors')
plt.plot(betas['Name'], betas['R_squared'], label = 'CAPM')
plt.title('Three Factor vs. CAPM R-Squared')
plt.xlabel('Company Name')
plt.ylabel('R-squared')
plt.legend(loc = 'upper left')
plt.xticks(rotation = 45)

data = pd.read_csv('/Users/noahconlon/Downloads/P Data Post 2014.csv')

data = data.drop(['vwretd', 'RET'], axis = 1)


data.columns = ['permno', 'date', 'name', 'Mkt-Rf', 'SMB', 'HML', 'RMW', 'CMA', 'Rf', 'Return']
data['Ret-Rf'] = data['Return']-data['Rf']
permno_post = list(dict.fromkeys(data['permno']))

betas_post = pd.DataFrame(permno_post, columns = ['permno']) 
vals_post = ['ION', 'QCOM','JACK', 'RES', 'NEURO', 'SEMPR', 'IlMNA', 'ACADIA', 'BIO', 'DCOM']
betas_post['FR_squared'] = np.nan
betas_post['Name'] = vals

for i in permno_post:
    subset_post = data[data['permno']==i].dropna()
    xpost = np.array(subset_post['Mkt-Rf'])
    x1post = sm.add_constant(np.array(subset_post[['Mkt-Rf','SMB','HML','RMW','CMA']]))
    ypost = np.array(subset_post['Ret-Rf'])
 
    modpost = sm.OLS(ypost,x1post).fit()

    betas_post.loc[betas_post['permno']==i, 'FR_squared'] = modpost.rsquared

dt = pd.read_csv('/Users/noahconlon/Downloads/P Data Pre 2014.csv')
dt = dt.drop(['vwretd', 'RET'], axis = 1)    
dt.columns = ['permno', 'date', 'name', 'Mkt-Rf', 'SMB', 'HML', 'RMW', 'CMA', 'Rf', 'Return']
dt['Ret-Rf'] = dt['Return']-dt['Rf']
permno_pre = list(dict.fromkeys(dt['permno']))

betas_pre = pd.DataFrame(permno_pre, columns = ['permno']) 
vals_pre = ['ION', 'QCOM','JACK', 'RES', 'NEURO', 'SEMPR', 'IlMNA', 'ACADIA', 'BIO', 'DCOM']

betas_pre['FR_squared'] = np.nan
betas_pre['Name'] = vals_pre

for i in permno_pre:
    subset_pre = dt[dt['permno']==i].dropna()
    xpre = np.array(subset_pre['Mkt-Rf'])
    x1pre = sm.add_constant(np.array(subset_pre[['Mkt-Rf','SMB','HML','RMW','CMA']]))
    ypre = np.array(subset_pre['Ret-Rf'])

    modpre = sm.OLS(ypre,x1pre).fit()

    betas_pre.loc[betas_pre['permno']==i, 'FR_squared'] = modpre.rsquared    

fig = plt.figure()
plt.plot(betas_pre['Name'], betas_pre['FR_squared'], label = 'Pre 2014')
plt.plot(betas_post['Name'], betas_post['FR_squared'], label = 'Post 2014')
plt.title('Pre 2014 vs. Post 2014')
plt.xlabel('Company Name')
plt.ylabel('R-squared')
plt.legend(loc = 'upper left')
plt.xticks(rotation = 45)

