#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import xarray as xr
import gc
import glob
import datetime
import dask.array as da
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


# In[2]:


#year = 2008
#ampm ='AM'
#sat = 'AIRS'
#time = '0130'
model_type = 'no5vars'  
plot_path = '/home/xchen/combined_LH_HR_Vert/plot/'

coef_SW = xr.open_dataset('/home/gmandorli/firstProject/extra_stuff/theoretical_irradiance/output/SW_reducing_factor.nc')
coef_SW_Jan = coef_SW.sel(month='Jan').SW_reducing_factor.mean('lat').mean('lon').values
#input_path = f'/homedata/xchen/combined_{model_type}_LH_HR_Vert/{year}/'  

HR_level = np.array([ 78.,  96., 119., 147., 181., 211., 
                     235., 262., 292., 325., 361., 402., 
                     448., 498., 555., 618., 688., 762., 
                     824., 875., 928., 984.]
                    )


# In[68]:



"""

Founctions definition Part

"""

"""
Read one-year combined LH_HR_Vert data 

"""

def read_data(year,sat,time,ampm,model_type):
    
    input_path = f'/homedata/xchen/combined_{model_type}_LH_HR_Vert/{year}/'   

    ds_list = []
    ds_input= []

    for month in range(1,1+1): 

        files = glob.glob(input_path+f'{month:02d}/LH_HR_Vert_{sat}_{time}{ampm}_*.nc') 
        ds_input = xr.open_mfdataset(files) 

        ds_list.append(ds_input)

    ds = xr.concat(ds_list,'time')
    
    
    # replace NAN of LH by "0".
    #data = ds['LH']
    #data = data.where(~data.isnull(), 0)
    #ds['LH'] = data.compute()
    
    #print(np.isnan(ds.LH.values).sum())
    #ds_scen0 = ds.where(ds['scen'] == 0, drop=True)
    #ds_scen1 = ds.where(ds['scen'] == 1, drop=True)
    #ds_scen2 = ds.where(ds['scen'] == 2, drop=True)
    #ds_scenall = ds.where(ds['scen'] >= 0, drop=True)
    
    #print('nb_scen0',len(ds_scen0.scen.values),'nb_scenall',len(ds_scenall.scen.values))# ds_scen0,ds_scen1,ds_scen2,ds_scenall
    
    
    return ds


def RR_class(ds,scen,option):
    
    # split data for different scenes, scen==1 for hgh clouds or scen==2 for mid-low clouds
    
    if scen != 'all':
        ds = ds.where(ds['scen'] == scen, drop=True)
    else:
        ds = ds
        
    if option == 'on':
    
        # define 3 different cases for: no rain, light rain, heavy rain
        ds_RReq0 = ds.where(ds['RainRate'] == 0, drop=True)
        ds_lt0gt2 = ds.where((ds['RainRate'] > 0) & (ds['RainRate'] < 2), drop=True)
        ds_RRgt2 = ds.where(ds['RainRate'] > 2, drop=True)

        ds_all = [ds_RReq0, ds_lt0gt2, ds_RRgt2]
       
        # dim reduction
        ds_LH=[]
        ds_LW=[]
        ds_SW=[]


        for n,data in enumerate(ds_all):
            ds_LH_list = data.LH.mean(dim=['time', 'lat', 'lon'])
            ds_LW_list = data.LW.mean(dim=['time', 'lat', 'lon'])
            ds_SW_list = data.SW.mean(dim=['time', 'lat', 'lon'])

            ds_LH.append(ds_LH_list)
            ds_LW.append(ds_LW_list)
            ds_SW.append(ds_SW_list)
        
    elif option == 'off':
        
        ds_LH = ds.LH.mean(dim=['time', 'lat', 'lon'])
        ds_LW = ds.LW.mean(dim=['time', 'lat', 'lon'])
        ds_SW = ds.SW.mean(dim=['time', 'lat', 'lon'])
        
        
    return ds_LH,ds_LW,ds_SW


def list_ds(year,scen,cut_option):
    
    ds_AIRS_AM = read_data(year,'AIRS','0130','AM','no5vars')
    ds_AIRS_PM = read_data(year,'AIRS','0130','PM','no5vars')
    ds_IASI_AM = read_data(year,"IASI",'0930','AM','no5vars')
    ds_IASI_PM = read_data(year,"IASI",'0930','PM','no5vars')
    
    # Create zeros_like data for SW
    ds_new = [0,0,0,0]
    all_datasets = [ds_AIRS_AM,ds_AIRS_PM,ds_IASI_AM,ds_IASI_PM]
    for i, ds in enumerate (all_datasets):
        if 'SW' not in ds.data_vars:
            ds_new[i] = ds.assign(SW=xr.zeros_like(ds['LW']))

        else:
            ds_new[i] = ds

    ds_AIRS_AM = ds_new[0]
    ds_AIRS_PM = ds_new[1]
    ds_IASI_AM = ds_new[2]
    ds_IASI_PM = ds_new[3]
    
    
    ds_LH_AIRS_AM,ds_LW_AIRS_AM,ds_SW_AIRS_AM = RR_class(ds_AIRS_AM,scen,cut_option)
    ds_LH_AIRS_PM,ds_LW_AIRS_PM,ds_SW_AIRS_PM = RR_class(ds_AIRS_PM,scen,cut_option)
    ds_LH_IASI_AM,ds_LW_IASI_AM,ds_SW_IASI_AM = RR_class(ds_IASI_AM,scen,cut_option)
    ds_LH_IASI_PM,ds_LW_IASI_PM,ds_SW_IASI_PM = RR_class(ds_IASI_PM,scen,cut_option)

    ds_LH_all = [ds_LH_AIRS_AM,ds_LH_AIRS_PM,ds_LH_IASI_AM,ds_LH_IASI_PM]
    ds_LW_all = [ds_LW_AIRS_AM,ds_LW_AIRS_PM,ds_LW_IASI_AM,ds_LW_IASI_PM]
    ds_SW_all = [ds_SW_AIRS_AM,ds_SW_AIRS_PM,ds_SW_IASI_AM,ds_SW_IASI_PM]
    
    
    # Too many variables! Wrap variables in a dictionary
    ds_info = {
        'ds_LH_all': ds_LH_all,
        'ds_LW_all': ds_LW_all,
        'ds_SW_all': ds_SW_all,
        'ds_LH_AIRS_AM': ds_LH_AIRS_AM,
        'ds_LW_AIRS_AM': ds_LW_AIRS_AM,
        'ds_SW_AIRS_AM': ds_SW_AIRS_AM,
        'ds_LH_AIRS_PM': ds_LH_AIRS_PM,
        'ds_LW_AIRS_PM': ds_LW_AIRS_PM,
        'ds_SW_AIRS_PM': ds_SW_AIRS_PM,
        'ds_LH_IASI_AM': ds_LH_IASI_AM,
        'ds_LW_IASI_AM': ds_LW_IASI_AM,
        'ds_SW_IASI_AM': ds_SW_IASI_AM,
        'ds_LH_IASI_PM': ds_LH_IASI_PM,
        'ds_LW_IASI_PM': ds_LW_IASI_PM,
        'ds_SW_IASI_PM': ds_SW_IASI_PM,
        
    }
    return ds_info



def reshape_clr_heat(ds_clr):
    # Variable name list to merge
    LW_names = []
    SW_names = []

    for i in range(22):
        var_LW = f'LW{i}'
        var_SW = f'SW{i}'

        LW_names.append(var_LW)
        SW_names.append(var_SW)

    if LW_names[0] in ds_clr.data_vars:
        ds_clr['LW'] = xr.concat([ds_clr[LW_name] for LW_name in  LW_names], dim='HR_level')
        ds_clr = ds_clr.drop_vars(LW_names)

    elif SW_names[0] in ds_clr.data_vars:

        #SW just exist for day-time, check it exists here or not!
        ds_clr['SW'] = xr.concat([ds_clr[SW_name] for SW_name in  SW_names], dim='HR_level')
        ds_clr = ds_clr.drop_vars(SW_names)

    #Fill in the new dimension：HR_level (do not drop the first 2 layers, cuz HR is a little bit higher than LH)
    ds_clr.update({'HR_level': HR_level}) 
    
    ds_clr = ds_clr.mean('lat').mean('lon')
    
    return ds_clr



def clr_heat(year,month):
    
    clrLW_AIRS_AM = xr.open_dataset(f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/{month}/clrskyLW_AIRS-ERAI_v8_0130AM_{year}{month}.nc')
    clrLW_AIRS_PM = xr.open_dataset(f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/{month}/clrskyLW_AIRS-ERAI_v8_0130PM_{year}{month}.nc')
    clrLW_IASI_AM = xr.open_dataset(f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/{month}/clrskyLW_IASI-ERAI_v8_0930AM_{year}{month}.nc')
    clrLW_IASI_PM = xr.open_dataset(f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/{month}/clrskyLW_IASI-ERAI_v8_0930PM_{year}{month}.nc')

    clrSW_AIRS_PM = xr.open_dataset(f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/{month}/clrskySW_AIRS-ERAI_v8_0130PM_{year}{month}.nc')
    clrSW_IASI_AM = xr.open_dataset(f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/{month}/clrskySW_IASI-ERAI_v8_0930AM_{year}{month}.nc')
    
   
    clrLW_AIRS_AM = reshape_clr_heat(clrLW_AIRS_AM)
    clrLW_AIRS_PM = reshape_clr_heat(clrLW_AIRS_PM)
    clrLW_IASI_AM = reshape_clr_heat(clrLW_IASI_AM)
    clrLW_IASI_PM = reshape_clr_heat(clrLW_IASI_PM)
    clrSW_AIRS_PM = reshape_clr_heat(clrSW_AIRS_PM)
    clrSW_IASI_AM = reshape_clr_heat(clrSW_IASI_AM)
    
    return clrLW_AIRS_AM,clrLW_AIRS_PM,clrLW_IASI_AM,clrLW_IASI_PM,clrSW_AIRS_PM,clrSW_IASI_AM



# Function to plot different heating profiles for different scenes, include LH,LW and SW
# !!! use this fuction only under the case that "cut_option == on"
def heat_profile_3cases(ds_all1,ds_all2,heat_type,scen):
    
    title = ['no rain','light rain','heavy rain']
    year = ['2008','2016']
    colors = ['#2878B5','#9AC9DB','#C82423','#F8AC8C'] 
    sat_time = ['AIRS AM','AIRS PM','IASI AM','IASI PM'] 
    lines = ['-','--']
    

    fig, axes = plt.subplots(figsize=(9,6),dpi=300,nrows=1, ncols=3,sharex=True, sharey=True)
    
    for l, ds_all in enumerate([ds_all1,ds_all2]):
    
        for i, ds in enumerate(ds_all):


            for n,ds_heat in enumerate(ds):
                if 'level' in ds_heat.dims:
                    Pressure = ds_heat.level.values
                elif 'HR_level' in ds_heat.dims:
                    Pressure = ds_heat.HR_level.values


                ax=axes[n]
                ax.plot(ds_heat.values,Pressure,label= f'{year[l]} {sat_time[i]}',linestyle=lines[l],color=colors[i]) 

                #ax.fill_betweenx(Pressure, 24*(mean_LH[m]+0.2*std_LH[m]), 24*(mean_LH[m]-0.2*std_LH[m]), facecolor=colors[m], alpha=0.2,edgecolor=colors[m],linestyle='--',linewidth=2) 

                ax.set_xlabel(f'{heat_type} (k/day)',fontsize=14,labelpad=14)
                ax.set_ylabel('Pressure (hPa)',fontsize=14)       
                ax.set_title (title[n], fontweight='bold',fontsize=14)
                ax.ticklabel_format(style='sci', scilimits=(1e-3,100))
                axes[0].legend(fontsize=9) #frameon=False
                ax.grid(b=True,axis='x')
                plt.tight_layout()
        
    # Invert the values of yaxis, cuz high (low) pressure corresponds to lower (upper) atmosphere.    
    ax.invert_yaxis()
    
    fig.savefig(plot_path + f'profile_combined_3cases_{model_type}_{heat_type}_scen{scen}.png')
    #plt.close()
        
    return 


# !!! use this fuction only under the case that "cut_option == on"
def LH_CRE_profile_3cases (ds_LH_AM_list,ds_LH_PM_list,ds_LW_AM_list,ds_LW_PM_list,ds_SW_AM_list,ds_SW_PM_list,sat,year,scen,):
    
    title = ['no rain','light rain','heavy rain']
    heat_type = ['LH','LH+HR']
    #colors = ['#2878B5','#9AC9DB','#C82423','#F8AC8C'] 
    #sat_time = ['AIRS AM','AIRS PM','IASI AM','IASI PM'] 
    lines = ['-','--']
    

    fig, axes = plt.subplots(figsize=(9,6),dpi=300,nrows=1, ncols=3,sharex=True, sharey=True)

    for i, (ds_LH_AM,ds_LH_PM,ds_LW_AM,ds_LW_PM,ds_SW_AM,ds_SW_PM) in enumerate(zip(ds_LH_AM_list,ds_LH_PM_list,ds_LW_AM_list,ds_LW_PM_list,ds_SW_AM_list,ds_SW_PM_list)):
    
            
        Pressure = ds_LH_AM.level.values

                
        CRE = 0.5*(ds_LW_AM.values + ds_LW_AM.values) + coef_SW_Jan * (ds_SW_AM.values+ds_SW_PM.values)   
        print('CRE: ',CRE[2:])
        
        LH = ds_LH_AM + ds_LH_PM
        print('LH: ',LH.values)
        
        LH_HR = LH + CRE[2:]
        
        print('LH + HR: ',LH_HR.values)


        ax=axes[i]
        ax.plot(LH_HR,Pressure,linestyle=lines[1],label= f'LH +CRE',color='#2878B5') 
        ax.plot(LH.values,Pressure,linestyle=lines[0],label= f'LH',color='#9AC9DB') 

        # ax.fill_betweenx(Pressure, 24*(mean_LH[m]+0.2*std_LH[m]), 24*(mean_LH[m]-0.2*std_LH[m]), facecolor=colors[m], alpha=0.2,edgecolor=colors[m],linestyle='--',linewidth=2) 

        ax.set_xlabel(f'heating (k/day)',fontsize=14,labelpad=14)
        ax.set_ylabel('Pressure (hPa)',fontsize=14)       
        ax.set_title (title[i], fontweight='bold',fontsize=14)
        ax.ticklabel_format(style='sci', scilimits=(1e-3,100))
        ax.grid(b=True,axis='x')
        axes[0].legend(fontsize=9) #frameon=False
            
        plt.tight_layout()
        
    # Invert the values of yaxis, cuz high (low) pressure corresponds to lower (upper) atmosphere.    
    ax.invert_yaxis()
    

    fig.savefig(plot_path + f'profile_combined_LH_CRE_3cases_{model_type}_{year}_{sat}_scen{scen}.png')
    #plt.close()
        
    return 


def LH_CRE_profile(ds_LH_AM,ds_LH_PM,ds_LW_AM,ds_LW_PM,ds_SW_AM,ds_SW_PM,clrLW_AM,clrLW_PM,clrSW,sat,year,scen):
    
    title = ['no rain','light rain','heavy rain']
    heat_type = ['LH','LH+HR']
    #colors = ['#2878B5','#9AC9DB','#C82423','#F8AC8C'] 
    #sat_time = ['AIRS AM','AIRS PM','IASI AM','IASI PM'] 
    lines = ['-','--']
    

    fig, axes = plt.subplots(figsize=(4,6),dpi=300,nrows=1, ncols=1,sharex=True, sharey=True)

    #for i, (ds_LH_AM,ds_LH_PM,ds_LW_AM,ds_LW_PM,ds_SW_AM,ds_SW_PM) in enumerate(zip(ds_LH_AM_list,ds_LH_PM_list,ds_LW_AM_list,ds_LW_PM_list,ds_SW_AM_list,ds_SW_PM_list)):
    
            
    Pressure = ds_LH_AM.level.values

    
    
    CRE = 0.5*(ds_LW_AM.values - clrLW_AM.LW.values + ds_LW_PM.values - clrLW_PM.LW.values) + coef_SW_Jan * (ds_SW_AM.values+ds_SW_PM.values-clrSW.SW.values)   
    print('CRE: ',CRE[2:])
        
    LH = ds_LH_AM + ds_LH_PM
    print('LH: ',LH.values)
        
    LH_HR = LH + CRE[2:]
        
    print('LH + HR: ',LH_HR.values)


    ax=axes
    ax.plot(LH_HR,Pressure,linestyle=lines[1],label= f'LH +CRE',color='#2878B5') 
    ax.plot(LH.values,Pressure,linestyle=lines[0],label= f'LH',color='#9AC9DB') 
    

    # ax.fill_betweenx(Pressure, 24*(mean_LH[m]+0.2*std_LH[m]), 24*(mean_LH[m]-0.2*std_LH[m]), facecolor=colors[m], alpha=0.2,edgecolor=colors[m],linestyle='--',linewidth=2) 
    ax.axvline(x=0, color='firebrick', linestyle='dotted') 
    ax.set_xlabel(f'heating (k/day)',fontsize=14,labelpad=14)
    ax.set_ylabel('Pressure (hPa)',fontsize=14)       
    #ax.set_title (title[i], fontweight='bold',fontsize=10)
    ax.ticklabel_format(style='sci', scilimits=(1e-3,100))
    ax.grid(b=True,axis='x')
    axes.legend(fontsize=9) #frameon=False
            
    plt.tight_layout()
        
    # Invert the values of yaxis, cuz high (low) pressure corresponds to lower (upper) atmosphere.    
    ax.invert_yaxis()
    

    fig.savefig(plot_path + f'profile_combined_LH_CRE_all_{model_type}_{year}_{sat}_scen{scen}.png')
    #plt.close()
        
    return 


def LH_map(ds,sat,ampm,year):

    fig4, ax4 = plt.subplots(3,1,figsize=(18,10),dpi=600,subplot_kw={'projection': ccrs.PlateCarree()})

    ds_high = ds.LH.sel(level=slice(100,324.595)).mean('level').mean('time')
    ds_mid = ds.LH.sel(level=slice(361.36,617.92)).mean('level').mean('time')
    ds_low = ds.LH.sel(level=slice(687.91,984.06)).mean('level').mean('time')
    
    #print(ds_high.LH.values.mean(),ds_mid.LH.values.mean(), ds_low.LH.values.mean())

    ds_map = [ds_low, ds_mid, ds_high]
    labels  = ['low-troposphere', 'mid-troposphere', 'high-troposphere']

    for n, data in enumerate(ds_map):
        
        im = data.plot(ax=ax4[n],vmin=-5,vmax=5,cmap='RdBu_r',add_colorbar=False) #vmin=-2,vmax=2
        plt.text(-0.06, 0.16, labels[n], fontsize=14, rotation=90, transform = ax4[n].transAxes) 

        ax4[n].add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=1, color='gray',alpha=0.8,linewidth=0.8)
        ax4[n].set_extent((data.lon[0], data.lon[-1], data.lat[0], data.lat[-1]),crs=ccrs.PlateCarree())
        gl = ax4[n].gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=1.2,color='gray',alpha=0.3,linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.xlabel_style = {'size':12,'rotation':0}
        gl.yformatter = LATITUDE_FORMATTER
        gl.ylabel_style = {'size':12,'rotation':0}

        #plt.tight_layout() 

    position = fig4.add_axes([0.94,0.15,0.018,0.7])
    cb = plt.colorbar(im,cax=position,orientation='vertical',extend='both')
    #cb.mappable.set_clim(-2,2)
    position.set_title('LH (K/data-day)', loc='center',fontsize=14,weight='normal')

    ax4[0].set_title(f'LH_map_{model_type}_{sat}_{ampm}_{year}', fontsize=20)
    #fig4.savefig(plot_path + f'LH_map_{model_type}_{sat}_{ampm}_{year}.png')
    #plt.close()
    
    return


# In[11]:


scen1 = 1
scen2 = 2
scen_all = 'all'


# In[12]:


ds_hgh_08_3cases = list_ds(2008,scen1,'on')
ds_hgh_16_3cases = list_ds(2016,scen1,'on')


# In[13]:


ds_mlow_08_3cases = list_ds(2008,scen2,'on')
ds_mlow_16_3cases = list_ds(2016,scen2,'on')


# In[48]:


heat_profile_3cases(ds_hgh_08_3cases['ds_LH_all'],ds_hgh_16_3cases['ds_LH_all'],'LH',scen1)


# In[49]:


heat_profile_3cases(ds_mlow_08_3cases['ds_LH_all'],ds_mlow_16_3cases['ds_LH_all'],'LH',scen2)


# In[50]:


heat_profile_3cases(ds_hgh_08_3cases['ds_LW_all'],ds_hgh_16_3cases['ds_LW_all'],'LW',scen1)


# In[51]:


heat_profile_3cases(ds_mlow_08_3cases['ds_LW_all'],ds_mlow_16_3cases['ds_LW_all'],'LW',scen2)


# In[52]:


heat_profile_3cases(ds_hgh_08_3cases['ds_SW_all'],ds_hgh_16_3cases['ds_SW_all'],'SW',scen1)


# In[53]:


heat_profile_3cases(ds_mlow_08_3cases['ds_SW_all'],ds_mlow_16_3cases['ds_SW_all'],'SW',scen2)


# In[54]:


ds_hgh_08 = list_ds(2008,scen1,'off')
ds_hgh_16 = list_ds(2016,scen1,'off')
ds_mlow_08= list_ds(2008,scen2,'off')
ds_mlow_16 = list_ds(2016,scen2,'off')
ds_all_08= list_ds(2008,scen_all,'off')
ds_all_16 = list_ds(2016,scen_all,'off')


# In[55]:


# read clr_sky LW and SW data:
clrLW_AIRS_AM_0801,clrLW_AIRS_PM_0801,clrLW_IASI_AM_0801,clrLW_IASI_PM_0801,clrSW_AIRS_PM_0801,clrSW_IASI_AM_0801 = clr_heat('2008','01')
clrLW_AIRS_AM_1601,clrLW_AIRS_PM_1601,clrLW_IASI_AM_1601,clrLW_IASI_PM_1601,clrSW_AIRS_PM_1601,clrSW_IASI_AM_1601 = clr_heat('2016','01')


# In[69]:


LH_CRE_profile(ds_all_08['ds_LH_AIRS_AM'],ds_all_08['ds_LH_AIRS_PM'],ds_all_08['ds_LW_AIRS_AM'],ds_all_08['ds_LW_AIRS_PM'],
                ds_all_08['ds_SW_AIRS_AM'],ds_all_08['ds_SW_AIRS_PM'],clrLW_AIRS_AM_0801,clrLW_AIRS_PM_0801,clrSW_AIRS_PM_0801,'AIRS',2008,scen_all)


# In[70]:


LH_CRE_profile(ds_hgh_08['ds_LH_AIRS_AM'],ds_hgh_08['ds_LH_AIRS_PM'],ds_hgh_08['ds_LW_AIRS_AM'],ds_hgh_08['ds_LW_AIRS_PM'],
                ds_hgh_08['ds_SW_AIRS_AM'],ds_hgh_08['ds_SW_AIRS_PM'],clrLW_AIRS_AM_0801,clrLW_AIRS_PM_0801,clrSW_AIRS_PM_0801,'AIRS',2008,scen1)


# In[71]:


LH_CRE_profile(ds_mlow_08['ds_LH_AIRS_AM'],ds_mlow_08['ds_LH_AIRS_PM'],ds_mlow_08['ds_LW_AIRS_AM'],ds_mlow_08['ds_LW_AIRS_PM'],
                ds_mlow_08['ds_SW_AIRS_AM'],ds_mlow_08['ds_SW_AIRS_PM'],clrLW_AIRS_AM_0801,clrLW_AIRS_PM_0801,clrSW_AIRS_PM_0801,'AIRS',2008,scen2)


# In[72]:


LH_CRE_profile(ds_all_16['ds_LH_AIRS_AM'],ds_all_16['ds_LH_AIRS_PM'],ds_all_16['ds_LW_AIRS_AM'],ds_all_16['ds_LW_AIRS_PM'],
                ds_all_16['ds_SW_AIRS_AM'],ds_all_16['ds_SW_AIRS_PM'],clrLW_AIRS_AM_1601,clrLW_AIRS_PM_1601,clrSW_AIRS_PM_1601,'AIRS',2016,scen_all)


# In[73]:


LH_CRE_profile(ds_hgh_16['ds_LH_AIRS_AM'],ds_hgh_16['ds_LH_AIRS_PM'],ds_hgh_16['ds_LW_AIRS_AM'],ds_hgh_16['ds_LW_AIRS_PM'],
                ds_hgh_16['ds_SW_AIRS_AM'],ds_hgh_16['ds_SW_AIRS_PM'],clrLW_AIRS_AM_1601,clrLW_AIRS_PM_1601,clrSW_AIRS_PM_1601,'AIRS',2016,scen1)


# In[74]:


print(f'This script needed {(datetime.datetime.now() - start_time).seconds} seconds') 


# In[ ]:




