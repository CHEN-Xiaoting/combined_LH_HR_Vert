#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import xarray as xr
import gc
import glob
import datetime
import matplotlib.dates as mdates
import itertools
import matplotlib.pyplot as plt 
#import seaborn as sns
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER,mticker

import warnings
warnings.filterwarnings("ignore")

start_time = datetime.datetime.now()


# In[31]:


plot_path = '/home/xchen/combined_LH_HR_Vert/plot/'


"""
Read one-year combined LH_HR_Vert data 

"""
year = 2008
ampm ='AM'
sat = 'AIRS'
time = '0130'

input_path = f'/homedata/xchen/combined_LH_HR_Vert/{year}/'  


ds_list = []
ds_input= []

for month in range(1,12+1): 
        
    files = glob.glob(input_path+f'{month:02d}/LH_HR_Vert_{sat}_{time}{ampm}_*.nc') 
    ds_input = xr.open_mfdataset(files) 

    ds_list.append(ds_input)
      
ds = xr.concat(ds_list,'time')


# In[16]:


ds_RReq0 = ds.where(ds['RainRate'] == 0, drop=True)
ds_lt0gt2 = ds.where((ds['RainRate'] > 0) & (ds['RainRate'] < 2), drop=True)
ds_RRgt2 = ds.where(ds['RainRate'] > 2, drop=True)


# In[18]:


ds_all = [ds_RReq0, ds_lt0gt2, ds_RRgt2]


# In[28]:


ds_LH=[]

for n,data in enumerate(ds_all):
    ds_list = data.LH.mean('time').mean('lat').mean('lon')
    ds_LH.append(ds_list)


# In[33]:


def LH_profile(ds_all):
    
    title = ['no rain','light rain','heavy rain']

    fig, axes = plt.subplots(figsize=(3*len(ds_all),6),dpi=300,nrows=1, ncols=len(ds_all),sharex=True, sharey=True)

    for n,ds in enumerate(ds_all):
        Pressure = ds.level.values
        
        ax=axes[n] 
        
        ax.plot(ds.values,Pressure) 

        #ax.fill_betweenx(Pressure, 24*(mean_LH[m]+0.2*std_LH[m]), 24*(mean_LH[m]-0.2*std_LH[m]), facecolor=colors[m], alpha=0.2,edgecolor=colors[m],linestyle='--',linewidth=2) 
        
        ax.set_xlabel('Latent heating (k/day)',fontsize=12,labelpad=12)
        ax.set_ylabel('Pressure (hPa)',fontsize=12)       
        ax.set_title (title[n], fontweight='bold',fontsize=10)
        ax.ticklabel_format(style='sci', scilimits=(1e-3,100))
        #ax.legend(fontsize=9)
        ax.grid(b=True,axis='x')
        plt.tight_layout()
        
    # Invert the values of yaxis, cuz high (low) pressure corresponds to lower (upper) atmosphere.    
    ax.invert_yaxis()
    
    fig.savefig(plot_path + f'profile_combined_LH_{sat}_{ampm}.png')
    #plt.close()
        
    return 

# In[]:

LH_map(ds)

LH_profile(ds_LH)


# In[7]:


print(f'This script needed {(datetime.datetime.now() - start_time).seconds} seconds')  


# In[ ]:




