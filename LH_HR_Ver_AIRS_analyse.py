#!/usr/bin/env python
# coding: utf-8

"""
Created on Fri May  12 15:42:31 2023 12 

@author: Xiaoting CHEN
"""

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
import matplotlib as mp
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER,mticker

import warnings
warnings.filterwarnings("ignore")

start_time = datetime.datetime.now()



#year = 2008
#ampm ='AM'
#sat = 'AIRS'
#time = '0130'
model_type = 'withfrac'  
plot_path = '/home/xchen/combined_LH_HR_Vert/plot/plot_AIRS/'

coef_SW = xr.open_dataset('/home/gmandorli/firstProject/extra_stuff/theoretical_irradiance/output/SW_reducing_factor.nc')
coef_SW_Jan = coef_SW.sel(month='Jan').SW_reducing_factor.mean('lat').mean('lon').values
#input_path = f'/homedata/xchen/combined_noclip_{model_type}_LH_HR_Vert/{year}/'  

HR_level = np.array([ 78.,  96., 119., 147., 181., 211., 
                     235., 262., 292., 325., 361., 402., 
                     448., 498., 555., 618., 688., 762., 
                     824., 875., 928., 984.]
                    )





"""

Founctions definition Part

"""

"""
Read combined LH_HR_Vert data 

"""

def read_data(year,sat,time,ampm,model_type):
    
    input_path = f'/homedata/xchen/combined_noclip_{model_type}_LH_HR_Vert/{year}/'   

    ds_list = []
    ds_input= []

    for month in range(1,1+1): 

        files = glob.glob(input_path+f'{month:02d}/LH_HR_Vert_{sat}_{time}{ampm}_*.nc') 
        ds_input = xr.open_mfdataset(files)
        
        ds_list.append(ds_input)

    ds = xr.concat(ds_list,'time')
    #ds['LH'] = ds['LH'].fillna(0)
    #ds = ds.fillna(0)
    
    return ds


def RR_class(ds,scen,option):
    
    # split data for different scenes, scen==1 for hgh clouds or scen==2 for mid-low clouds
    
    if scen != 'all':
        ds = ds.where(ds['scen'] == scen, drop=True)
        #print(ds.LH.values)
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
    
    ds_AIRS_AM = read_data(year,'AIRS','0130','AM',model_type)
    ds_AIRS_PM = read_data(year,'AIRS','0130','PM',model_type)
    ds_IASI_AM = read_data(year,"IASI",'0930','AM',model_type)
    ds_IASI_PM = read_data(year,"IASI",'0930','PM',model_type)
    
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
        'ds_AIRS_AM': ds_AIRS_AM,
        'ds_AIRS_PM': ds_AIRS_PM,
        'ds_IASI_AM': ds_IASI_AM,
        'ds_IASI_PM': ds_IASI_PM,
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
    
    ds_clr = ds_clr.interpolate_na(dim="lon", method="nearest", fill_value="extrapolate")
    
    ds_clr_map = ds_clr
    
    ds_clr = ds_clr.mean('lat').mean('lon')
    
    return ds_clr,ds_clr_map



def clr_heat(year,month):
    
    # !!! Need to change this part to read all data afterwards !!! Here only for Jan of 2008 and 2016
    clrLW_AIRS_AM = xr.open_dataset(f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/{month}/clrskyLW_AIRS-ERAI_v8_0130AM_{year}{month}.nc')
    clrLW_AIRS_PM = xr.open_dataset(f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/{month}/clrskyLW_AIRS-ERAI_v8_0130PM_{year}{month}.nc')
    clrLW_IASI_AM = xr.open_dataset(f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/{month}/clrskyLW_IASI-ERAI_v8_0930AM_{year}{month}.nc')
    clrLW_IASI_PM = xr.open_dataset(f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/{month}/clrskyLW_IASI-ERAI_v8_0930PM_{year}{month}.nc')

    clrSW_AIRS_PM = xr.open_dataset(f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/{month}/clrskySW_AIRS-ERAI_v8_0130PM_{year}{month}.nc')
    clrSW_IASI_AM = xr.open_dataset(f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/{month}/clrskySW_IASI-ERAI_v8_0930AM_{year}{month}.nc')
    
   
    clrLW_AIRS_AM,clrLW_AIRS_AM_map = reshape_clr_heat(clrLW_AIRS_AM)
    clrLW_AIRS_PM,clrLW_AIRS_PM_map = reshape_clr_heat(clrLW_AIRS_PM)
    clrLW_IASI_AM,clrLW_IASI_AM_map = reshape_clr_heat(clrLW_IASI_AM)
    clrLW_IASI_PM,clrLW_IASI_PM_map = reshape_clr_heat(clrLW_IASI_PM)
    clrSW_AIRS_PM,clrSW_AIRS_PM_map = reshape_clr_heat(clrSW_AIRS_PM)
    clrSW_IASI_AM,clrSW_IASI_AM_map = reshape_clr_heat(clrSW_IASI_AM)
    
    clr_info = {
        'clrLW_AIRS_AM': clrLW_AIRS_AM,
        'clrLW_AIRS_PM': clrLW_AIRS_PM,
        'clrLW_IASI_AM': clrLW_IASI_AM,
        'clrLW_IASI_PM': clrLW_IASI_PM,
        'clrSW_AIRS_PM': clrSW_AIRS_PM,
        'clrSW_AIRS_PM': clrSW_AIRS_PM,
        'clrSW_IASI_AM': clrSW_IASI_AM,

        'clrLW_AIRS_AM_map': clrLW_AIRS_AM_map,
        'clrLW_AIRS_PM_map': clrLW_AIRS_PM_map,
        'clrLW_IASI_AM_map': clrLW_IASI_AM_map,
        'clrLW_IASI_PM_map': clrLW_IASI_PM_map,
        'clrSW_AIRS_PM_map': clrSW_AIRS_PM_map,
        'clrSW_AIRS_PM_map': clrSW_AIRS_PM_map,
        'clrSW_IASI_AM_map': clrSW_IASI_AM_map,
        
    }
    return clr_info



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
    
    fig.savefig(plot_path + f'profile_combined_3cases_AIRS_{model_type}_{heat_type}_scen{scen}.png')
    #plt.close()
        
    return 


# !!! use this fuction only under the case that "cut_option == on"
def LH_CRE_profile_3cases (ds_LH_AM_list,ds_LH_PM_list,ds_LW_AM_list,ds_LW_PM_list,ds_SW_AM_list,ds_SW_PM_list,sat,year,scen,):
    
    title = ['no rain','light rain','heavy rain']
    heat_type = ['LH','LH+HR']
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
    

    #fig.savefig(plot_path + f'profile_combined_LH_CRE_3cases_{model_type}_{year}_{sat}_scen{scen}.png')
    #plt.close()
        
    return 


def LH_CRE_profile(ds_LH_AM,ds_LH_PM,ds_LW_AM,ds_LW_PM,ds_SW_AM,ds_SW_PM,clrLW_AM,clrLW_PM,clrSW,sat,year,scen):
    
    title = ['no rain','light rain','heavy rain']
    heat_type = ['LH','LH+HR']
    lines = ['-','--']
    

    fig, axes = plt.subplots(figsize=(4,6),dpi=300,nrows=1, ncols=1,sharex=True, sharey=True)
        
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
    

    #fig.savefig(plot_path + f'profile_combined_LH_CRE_all_{model_type}_{year}_{sat}_scen{scen}.png')
    plt.close()
        
    return 


def LH_CRE_profile_0816(ds_LH_AM_08,ds_LH_PM_08,ds_LW_AM_08,ds_LW_PM_08,ds_SW_AM_08,ds_SW_PM_08,ds_LH_AM_16,ds_LH_PM_16,ds_LW_AM_16,ds_LW_PM_16,ds_SW_AM_16,ds_SW_PM_16,clrLW_AM,clrLW_PM,clrSW,sat,scen):
    
    title = ['no rain','light rain','heavy rain']
    heat_type = ['LH','LH+HR']
    lines = ['-','--']
    

    fig, axes = plt.subplots(figsize=(8,6),dpi=300,nrows=1, ncols=1,sharex=True, sharey=True)
    
            
    Pressure = ds_LH_AM_08.level.values

    
    CRE_08 = 0.5*(ds_LW_AM_08.values - clrLW_AM.LW.values + ds_LW_PM_08.values - clrLW_PM.LW.values) + coef_SW_Jan * (ds_SW_AM_08.values+ds_SW_PM_08.values-clrSW.SW.values)   
    CRE_16 = 0.5*(ds_LW_AM_16.values - clrLW_AM.LW.values + ds_LW_PM_16.values - clrLW_PM.LW.values) + coef_SW_Jan * (ds_SW_AM_16.values+ds_SW_PM_16.values-clrSW.SW.values)  
    #print('CRE: ',CRE_08[2:])
        
    LH_08 = ds_LH_AM_08 + ds_LH_PM_08
    LH_16 = ds_LH_AM_16 + ds_LH_PM_16
    #print('LH: ',LH.values)
        
    LH_HR_08 = LH_08 + CRE_08[2:]
    LH_HR_16 = LH_16 + CRE_16[2:]
        
    #print('LH + HR: ',LH_HR.values)


    ax=axes
    ax.plot(LH_HR_08,Pressure,linestyle=lines[1],label= f'2008 LH +CRE',lw=3,color='#2878B5') 
    ax.plot(LH_08.values,Pressure,linestyle=lines[0],label= f'2008 LH',lw=3,color='#9AC9DB') 
    
    ax.plot(LH_HR_16,Pressure,linestyle=lines[1],label= f'2016 LH +CRE',lw=3,color='#F8AC8C') 
    ax.plot(LH_16.values,Pressure,linestyle=lines[0],label= f'2016 LH',lw=3,color='#C82423') 
    

    # ax.fill_betweenx(Pressure, 24*(mean_LH[m]+0.2*std_LH[m]), 24*(mean_LH[m]-0.2*std_LH[m]), facecolor=colors[m], alpha=0.2,edgecolor=colors[m],linestyle='--',linewidth=2) 
    ax.axvline(x=0, color='black', linestyle='dotted') 
    ax.set_xlabel(f'heating (k/day)',fontsize=18,labelpad=18)
    ax.set_ylabel('Pressure (hPa)',fontsize=18,labelpad=18)       
    #ax.set_title (title[i], fontweight='bold',fontsize=10)
    ax.ticklabel_format(style='sci', scilimits=(1e-3,100))
    ax.tick_params(labelsize=16)
    ax.grid(b=True,axis='x')
    axes.legend(fontsize=12) #frameon=False
            
    plt.tight_layout()
        
    # Invert the values of yaxis, cuz high (low) pressure corresponds to lower (upper) atmosphere.    
    ax.invert_yaxis()
    

    #fig.savefig(plot_path + f'profile_combined_LH_CRE_all_{model_type}_{sat}_scen{scen}_0816.png')
    plt.close()
        
    return


def LH_map_0816(ds_AM,ds_PM,P_min,P_max,sat,ampm,year):


    ds_mid_AM = ds_AM.LH.sel(level=slice(P_min,P_max)).mean(['level','time'])
    ds_mid_PM = ds_PM.LH.sel(level=slice(P_min,P_max)).mean(['level','time'])
    
    ds_mid_day = ds_mid_AM + ds_mid_PM
    
    
    if ampm == 'AM':
        
        lon = ds_mid_AM.lon.values
        lat = ds_mid_AM.lat.values
        values = ds_mid_AM.values
        
    elif ampm == 'PM':
        
        lon = ds_mid_PM.lon.values
        lat = ds_mid_PM.lat.values
        values = ds_mid_PM.values

        
    elif ampm == 'day':    
        
        lon = ds_mid_day.lon.values
        lat = ds_mid_day.lat.values
        values = ds_mid_day.values
    
    fig = plt.figure(figsize=(18, 8),dpi = 600)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    
    im = ax.contourf(lon, lat, values,vmin=-24,vmax=24, transform=ccrs.PlateCarree(), cmap='RdBu_r',add_colorbar=True)
   
    ax.add_feature(cfeat.LAND, facecolor='0.9', edgecolor='black')
    #ax.add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=1, color='gray', edgecolor='black',alpha=0.8,linewidth=0.8)
    ax.add_feature(cfeat.COASTLINE, edgecolor='black')
    ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')

    # Set longitude and latitude tick labels
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree(central_longitude=180))
    ax.set_xticklabels(['0', '60$^\circ$E', '120$^\circ$E', '180$^\circ$E', '120$^\circ$W', '60$^\circ$W', '0'], fontsize=12)
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_yticklabels(['20$^\circ$S', '10$^\circ$S', '0','10$^\circ$N', '20$^\circ$N'], fontsize=12)  
 

    position = fig.add_axes([0.16,0.27,0.7,0.02])
    
    #cbar = fig.colorbar(im, cax=position, orientation='horizontal',extend='both')
    #cbar.ax.set_title('LH', fontsize=12, pad=-30)
    
    cb = plt.colorbar(im,cax=position,orientation='horizontal',extend='both')
    #cb.set_clim(-5,35)
    position.set_title('LH (K/data-day)', loc='center',fontsize=14,weight='normal')

    ax.set_title(f'LH {sat} {year} {ampm} ({P_min} - {P_max} hPa)', fontsize=16,fontweight='bold',pad=10)
    #fig.savefig(plot_path + f'LH_map_{model_type}_{P_min}_{P_max}_{sat}_{ampm}_{year}.png')
    #plt.close()
    
    return


def CRE_map_0816(ds_AM,ds_PM,clrLW_AM,clrLW_PM,clrSW,P_min,P_max,sat,ampm,year):


    ds_LH_AM = ds_AM.LH.sel(level=slice(P_min,P_max)).mean(['level','time']) 
    ds_LH_PM = ds_PM.LH.sel(level=slice(P_min,P_max)).mean(['level','time'])

    
    ds_LW_AM = ds_AM.LW.sel(HR_level=slice(P_min,P_max)).mean(['HR_level','time'])
    ds_LW_PM = ds_PM.LW.sel(HR_level=slice(P_min,P_max)).mean(['HR_level','time'])

    
    ds_SW_AM = ds_AM.SW.sel(HR_level=slice(P_min,P_max)).mean(['HR_level','time'])
    ds_SW_PM = ds_PM.SW.sel(HR_level=slice(P_min,P_max)).mean(['HR_level','time'])

    
    
    ds_LH_day = ds_LH_AM + ds_LH_PM
    ds_LW_day = ds_LW_AM + ds_LW_PM

    
    clrLW_AM = clrLW_AM.LW.mean('HR_level')
    clrLW_PM = clrLW_PM.LW.mean('HR_level')
    clrSW = clrSW.SW.mean('HR_level')

    
    CRE = 0.5*(ds_LW_AM - clrLW_AM + ds_LW_PM - clrLW_PM) + coef_SW_Jan * (ds_SW_AM+ds_SW_PM - clrSW)
    
    CRE_AM = ds_LW_AM - clrLW_AM
    CRE_PM = ds_LW_PM - clrLW_PM + coef_SW_Jan * (ds_SW_PM - clrSW)  
    
    
    if ampm == 'AM':
    
        lon = CRE_AM.lon.values
        lat = CRE_AM.lat.values
        values = CRE_AM.values #np.isnan(CRE_AM.values)

        
    elif ampm == 'PM':
        
        lon = CRE_PM.lon.values
        lat = CRE_PM.lat.values
        values = CRE_PM.values 
        
    elif ampm == 'day':

        lon = CRE.lon.values
        lat = CRE.lat.values
        values = CRE.values 

           
    
    fig = plt.figure(figsize=(18, 8),dpi = 600)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))    
    
        
    im = ax.contourf(lon, lat, values,vmin=-1.6,vmax=1.6, transform=ccrs.PlateCarree(), cmap='RdBu_r',add_colorbar=True)
   
    ax.add_feature(cfeat.LAND, facecolor='0.9', edgecolor='black')
    #ax.add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=1, color='gray', edgecolor='black',alpha=0.8,linewidth=0.8)
    ax.add_feature(cfeat.COASTLINE, edgecolor='black')
    ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    # Set longitude and latitude tick labels
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree(central_longitude=180))
    ax.set_xticklabels(['0', '60$^\circ$E', '120$^\circ$E', '180$^\circ$E', '120$^\circ$W', '60$^\circ$W', '0'], fontsize=12)
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_yticklabels(['20$^\circ$S', '10$^\circ$S', '0','10$^\circ$N', '20$^\circ$N'], fontsize=12)  
    
    position = fig.add_axes([0.16,0.27,0.7,0.02])
    
    cb = plt.colorbar(im,cax=position,orientation='horizontal',extend='both')
    cb.mappable.set_clim(-1.6,1.6)
    #cb.mappable.set_clim(values.min(), values.max())
    position.set_title('CRE (K/data-day)', loc='center',fontsize=14,weight='normal')

    ax.set_title(f'CRE {sat} {year} {ampm} ({P_min} - {P_max} hPa)',fontsize=16,fontweight='bold')
    #fig.savefig(plot_path + f'CRE_map_{model_type}_{P_min}_{P_max}_{sat}_{ampm}_{year}.png')
    #plt.close()
    
    return


def LH_map_diff_0816(ds_AM_08,ds_PM_08,ds_AM_16,ds_PM_16,P_min,P_max,sat,ampm,year):


    ds_mid_AM_08 = ds_AM_08.LH.sel(level=slice(P_min,P_max)).mean(['level','time'])  #324.595,554.9
    ds_mid_PM_08 = ds_PM_08.LH.sel(level=slice(P_min,P_max)).mean(['level','time'])
    ds_mid_AM_16 = ds_AM_16.LH.sel(level=slice(P_min,P_max)).mean(['level','time'])
    ds_mid_PM_16 = ds_PM_16.LH.sel(level=slice(P_min,P_max)).mean(['level','time'])
    
    ds_mid_day_08 = ds_mid_AM_08 + ds_mid_PM_08
    ds_mid_day_16 = ds_mid_AM_16 + ds_mid_PM_16
    
    ds_AM_diff = ds_mid_AM_16-ds_mid_AM_08
    ds_PM_diff = ds_mid_PM_16-ds_mid_PM_08
    ds_day_iff = ds_mid_day_16-ds_mid_day_08
    
    if ampm == 'AM':
    
        lon = ds_AM_diff.lon.values
        lat = ds_AM_diff.lat.values
        values = ds_AM_diff.values
        
    elif ampm == 'PM':
        
        lon = ds_PM_diff.lon.values
        lat = ds_PM_diff.lat.values
        values = ds_PM_diff.values
        
    elif ampm == 'day':
        
        lon = ds_day_diff.lon.values
        lat = ds_day_diff.lat.values
        values = ds_day_diff.values
        
    
    fig = plt.figure(figsize=(18, 8),dpi = 600)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))    
    
        
    im = ax.contourf(lon, lat, values,vmin=-120,vmax=120, transform=ccrs.PlateCarree(), cmap='RdBu_r',add_colorbar=True)
   
    ax.add_feature(cfeat.LAND, facecolor='0.9', edgecolor='black')
    #ax.add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=1, color='gray', edgecolor='black',alpha=0.8,linewidth=0.8)
    ax.add_feature(cfeat.COASTLINE, edgecolor='black')
    ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    # Set longitude and latitude tick labels
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree(central_longitude=180))
    ax.set_xticklabels(['0', '60$^\circ$E', '120$^\circ$E', '180$^\circ$E', '120$^\circ$W', '60$^\circ$W', '0'], fontsize=12)
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_yticklabels(['20$^\circ$S', '10$^\circ$S', '0','10$^\circ$N', '20$^\circ$N'], fontsize=12)  
    
    position = fig.add_axes([0.16,0.27,0.7,0.02])
    
    
    cb = plt.colorbar(im,cax=position,orientation='horizontal',extend='both')
    #cb.mappable.set_clim(-5,5)
    position.set_title('LH (K/data-day)', loc='center',fontsize=14,weight='normal')

    ax.set_title(f'LH {sat} 2016 - 2008 {ampm} ({P_min} - {P_max} hPa)',fontsize=16,fontweight='bold')
    #fig.savefig(plot_path + f'LH_map_{model_type}_{P_min}_{P_max}_{sat}_{ampm}_{year}_diff.png')
    #plt.close()
    
    return


def CRE_map_diff_0816(ds_AM_08,ds_PM_08,ds_AM_16,ds_PM_16,clrLW_AM_08,clrLW_PM_08,clrSW_08,clrLW_AM_16,clrLW_PM_16,clrSW_16,P_min,P_max,sat,ampm,year):


    ds_LH_AM_08 = ds_AM_08.LH.sel(level=slice(P_min,P_max)).mean(['level','time']) 
    ds_LH_PM_08 = ds_PM_08.LH.sel(level=slice(P_min,P_max)).mean(['level','time'])
    ds_LH_AM_16 = ds_AM_16.LH.sel(level=slice(P_min,P_max)).mean(['level','time'])
    ds_LH_PM_16 = ds_PM_16.LH.sel(level=slice(P_min,P_max)).mean(['level','time'])
    
    ds_LW_AM_08 = ds_AM_08.LW.sel(HR_level=slice(P_min,P_max)).mean(['HR_level','time'])
    ds_LW_PM_08 = ds_PM_08.LW.sel(HR_level=slice(P_min,P_max)).mean(['HR_level','time'])
    ds_LW_AM_16 = ds_AM_16.LW.sel(HR_level=slice(P_min,P_max)).mean(['HR_level','time'])
    ds_LW_PM_16 = ds_PM_16.LW.sel(HR_level=slice(P_min,P_max)).mean(['HR_level','time'])
    
    ds_SW_AM_08 = ds_AM_08.SW.sel(HR_level=slice(P_min,P_max)).mean(['HR_level','time'])
    ds_SW_PM_08 = ds_PM_08.SW.sel(HR_level=slice(P_min,P_max)).mean(['HR_level','time'])
    ds_SW_AM_16 = ds_AM_16.SW.sel(HR_level=slice(P_min,P_max)).mean(['HR_level','time'])
    ds_SW_PM_16 = ds_PM_16.SW.sel(HR_level=slice(P_min,P_max)).mean(['HR_level','time'])
    
    
    ds_LH_day_08 = ds_LH_AM_08 + ds_LH_PM_08
    ds_LH_day_16 = ds_LH_AM_16 + ds_LH_PM_16
    
    ds_LW_day_08 = ds_LW_AM_08 + ds_LW_PM_08
    ds_LW_day_16 = ds_LW_AM_16 + ds_LW_PM_16
    
    
    clrLW_AM_08 = clrLW_AM_08.LW.mean('HR_level')
    clrLW_PM_08 = clrLW_PM_08.LW.mean('HR_level')
    clrSW_08 = clrSW_08.SW.mean('HR_level')
    clrLW_AM_16 = clrLW_AM_16.LW.mean('HR_level')
    clrLW_PM_16 = clrLW_PM_16.LW.mean('HR_level')
    clrSW_16 = clrSW_16.SW.mean('HR_level')
    
    CRE_08 = 0.5*(ds_LW_AM_08 - clrLW_AM_08 + ds_LW_PM_08 - clrLW_PM_08) + coef_SW_Jan * (ds_SW_AM_08+ds_SW_PM_08 - clrSW_08)   
    CRE_16 = 0.5*(ds_LW_AM_16 - clrLW_AM_16 + ds_LW_PM_16 - clrLW_PM_16) + coef_SW_Jan * (ds_SW_AM_16+ds_SW_PM_16 - clrSW_16)
    
    
    ds_LH_diff = ds_LH_day_16 - ds_LH_day_08
    CRE_diff = CRE_16 - CRE_08
    
    CRE_08_AM = ds_LW_AM_08 - clrLW_AM_08  
    CRE_16_AM = ds_LW_AM_16 - clrLW_AM_16 
    
    CRE_08_PM = ds_LW_PM_08 - clrLW_PM_08  + coef_SW_Jan * (ds_SW_PM_08 - clrSW_08)   
    CRE_16_PM = ds_LW_PM_16 - clrLW_PM_16 + coef_SW_Jan * (ds_SW_PM_16 - clrSW_16)
    
    CRE_AM_diff = CRE_16_AM - CRE_08_AM
    CRE_PM_diff = CRE_16_PM - CRE_08_PM
    
    #print('CRE_08: ',CRE_08[2:].values)
    if ampm == 'AM':
        
        lon = CRE_AM_diff.lon.values
        lat = CRE_AM_diff.lat.values
        values = CRE_AM_diff.values
    
    elif ampm == 'PM':
    
        lon = CRE_PM_diff.lon.values
        lat = CRE_PM_diff.lat.values
        values = CRE_PM_diff.values
        
    elif ampm == 'day':
        
        lon = CRE_diff.lon.values
        lat = CRE_diff.lat.values
        values = CRE_diff.values
        
    
    fig = plt.figure(figsize=(18, 8),dpi = 600)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))    
    
        
    im = ax.contourf(lon, lat, values,vmin=-1.5,vmax=1.5, transform=ccrs.PlateCarree(), cmap='RdBu_r',add_colorbar=True)
   
    ax.add_feature(cfeat.LAND, facecolor='0.9', edgecolor='black')
    #ax.add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=1, color='gray', edgecolor='black',alpha=0.8,linewidth=0.8)
    ax.add_feature(cfeat.COASTLINE, edgecolor='black')
    ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    # Set longitude and latitude tick labels
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree(central_longitude=180))
    ax.set_xticklabels(['0', '60$^\circ$E', '120$^\circ$E', '180$^\circ$E', '120$^\circ$W', '60$^\circ$W', '0'], fontsize=12)
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_yticklabels(['20$^\circ$S', '10$^\circ$S', '0','10$^\circ$N', '20$^\circ$N'], fontsize=12)  
    
    position = fig.add_axes([0.16,0.27,0.7,0.02])

    cb = plt.colorbar(im,cax=position,orientation='horizontal',extend='both')
    cb.mappable.set_clim(-1.5,1.5)

    position.set_title('CRE (K/data-day)', loc='center',fontsize=14,weight='normal')

    ax.set_title(f'CRE {sat} 2016 - 2008 {ampm} ({P_min} - {P_max} hPa)',fontsize=16,fontweight='bold')
    #fig.savefig(plot_path + f'CRE_map_{model_type}_{P_min}_{P_max}_{sat}_{ampm}_{year}_diff.png')
    #plt.close()
    
    return


def CRE_map_0816_clrsky(clrLW_AM,clrLW_PM,clrSW,P_min,P_max,sat,ampm,year):
    
    clrLW_AM = clrLW_AM.LW.mean('HR_level')
    clrLW_PM = clrLW_PM.LW.mean('HR_level')
    clrSW = clrSW.SW.mean('HR_level')

    
    CRE = 0.5*(clrLW_AM + clrLW_PM) + coef_SW_Jan * clrSW
    CRE_AM = clrLW_AM   
    CRE_PM = clrLW_PM + coef_SW_Jan * clrSW 
    
    
    if ampm == 'AM':
        
        lon = CRE_AM.lon.values
        lat = CRE_AM.lat.values
        values = CRE_AM.values
        #print(np.isnan(values).sum())
        
    elif ampm == 'PM':
        
        lon = CRE_PM.lon.values
        lat = CRE_PM.lat.values
        values = CRE_PM.values
        #print(np.isnan(values).sum())
        
    elif ampm == 'day':
        lon = CRE.lon.values
        lat = CRE.lat.values
        values = CRE.values #np.isnan(CRE.values)
        #print(np.isnan(values).sum())
        
    
    fig = plt.figure(figsize=(18, 8),dpi = 600)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))    
    
         
    im = ax.contourf(lon, lat, values,vmin=-3,vmax=3, transform=ccrs.PlateCarree(), cmap='RdBu_r',add_colorbar=True) #vmin=-3,vmax=1.6
    
    cmap = im.get_cmap()
    cmap.set_bad(color='white')
    im.set_cmap(cmap)
   
    ax.add_feature(cfeat.LAND, facecolor='0.9', edgecolor='black')
    #ax.add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=1, color='gray', edgecolor='black',alpha=0.8,linewidth=0.8)
    ax.add_feature(cfeat.COASTLINE, edgecolor='black')
    ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
    # Set longitude and latitude tick labels
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree(central_longitude=180))
    ax.set_xticklabels(['0', '60$^\circ$E', '120$^\circ$E', '180$^\circ$E', '120$^\circ$W', '60$^\circ$W', '0'], fontsize=12)
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_yticklabels(['20$^\circ$S', '10$^\circ$S', '0','10$^\circ$N', '20$^\circ$N'], fontsize=12)  
    
    position = fig.add_axes([0.16,0.27,0.7,0.02])

    cb = plt.colorbar(im,cax=position,orientation='horizontal',extend='both')
    #cb.mappable.set_clim(-1.6,1.6)
    position.set_title('CRE (K/data-day)', loc='center',fontsize=14,weight='normal')

    ax.set_title(f'CRE clrsky {sat} {year} {ampm}  ({P_min} - {P_max} hPa)',fontsize=16,fontweight='bold')
    #fig.savefig(plot_path + f'CRE_map_{model_type}_{P_min}_{P_max}_{sat}_{ampm}_{year}_clrsky.png')
    #plt.close()
    
    return


def LH_sys_map_0816(ds_AM,ds_PM,ds_sys_AM,ds_sys_PM,id,P_min,P_max,sat,ampm,year):


    ds_mid_AM = ds_AM.LH.sel(level=slice(P_min,P_max)).mean('level')#.mean('time')
    ds_mid_PM = ds_PM.LH.sel(level=slice(P_min,P_max)).mean('level')#.mean('time')
    
    ds_mid_day = ds_mid_AM + ds_mid_PM
    ds_sys_day = ds_sys_AM + ds_sys_PM
    
    #print(ds_mid_day.time)
    #print(ds_sys_day.time)
    
    
    if ampm=='AM':
        
        lon = ds_mid_AM.lon.values
        lat = ds_mid_AM.lat.values
        
         #values = xr.where(ds_sys_AM.idc > id, ds_mid_AM, np.nan) #in this way,dimension time not equal
        values = ds_mid_AM.where(ds_sys_AM.idc.values>id, np.nan)
        values = values.mean('time')
        
        
    elif ampm=='PM':
        
        lon = ds_mid_PM.lon.values
        lat = ds_mid_PM.lat.values
        values = ds_mid_PM.where(ds_sys_PM.idc.values>id, np.nan)
        values = values.mean('time')
        
    elif ampm=='day':
        
        lon = ds_mid_day.lon.values
        lat = ds_mid_day.lat.values
        values = ds_mid_day.where(ds_sys_day.idc.values>id, np.nan)
        values = values.mean('time')
        

    
    fig = plt.figure(figsize=(18, 8),dpi = 500)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    
    im = ax.contourf(lon, lat, values,vmin=-43,vmax=43, transform=ccrs.PlateCarree(), cmap='RdBu_r',add_colorbar=True)
   
    ax.add_feature(cfeat.LAND, facecolor='0.9', edgecolor='black')
    #ax.add_feature(cfeat.COASTLINE.with_scale('50m'), zorder=1, color='gray', edgecolor='black',alpha=0.8,linewidth=0.8)
    ax.add_feature(cfeat.COASTLINE, edgecolor='black')
    ax.gridlines(draw_labels=False, linewidth=1, color='gray', alpha=0.5, linestyle='--')

    # Set longitude and latitude tick labels
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree(central_longitude=180))
    ax.set_xticklabels(['0', '60$^\circ$E', '120$^\circ$E', '180$^\circ$E', '120$^\circ$W', '60$^\circ$W', '0'], fontsize=12)
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.set_yticklabels(['20$^\circ$S', '10$^\circ$S', '0','10$^\circ$N', '20$^\circ$N'], fontsize=12)  
 

    position = fig.add_axes([0.16,0.27,0.7,0.02])
    
    
    cb = plt.colorbar(im,cax=position,orientation='horizontal',extend='both')
    #cb.set_clim(-5,35)
    
    position.set_title('LH (K/data-day)', loc='center',fontsize=14,weight='normal')

    ax.set_title(f'LH - cls systems {sat} {year} {ampm}   ({P_min} - {P_max} hPa)', fontsize=16,fontweight='bold',pad=10)
    #fig.savefig(plot_path + f'LH_map_{model_type}_{P_min}_{P_max}_{sat}_{ampm}_{year}.png')
    #plt.close()
    
    return


# In[4]:


scen0 = 0
scen1 = 1
scen2 = 2
scen_all = 'all'


# In[ ]:


ds_clr_08_3cases = list_ds(2008,scen0,'on')
ds_clr_16_3cases = list_ds(2016,scen0,'on')


# In[ ]:


ds_hgh_08_3cases = list_ds(2008,scen1,'on')
ds_hgh_16_3cases = list_ds(2016,scen1,'on')


# In[ ]:


ds_mlow_08_3cases = list_ds(2008,scen2,'on')
ds_mlow_16_3cases = list_ds(2016,scen2,'on')


# In[ ]:


ds_clr_08 = list_ds(2008,scen0,'off')
ds_clr_16 = list_ds(2016,scen0,'off')
ds_hgh_08 = list_ds(2008,scen1,'off')
ds_hgh_16 = list_ds(2016,scen1,'off')
ds_mlow_08= list_ds(2008,scen2,'off')
ds_mlow_16 = list_ds(2016,scen2,'off')
ds_all_08= list_ds(2008,scen_all,'off')
ds_all_16 = list_ds(2016,scen_all,'off')


# In[ ]:


clr_info_08 = clr_heat('2008','01')
clr_info_16 = clr_heat('2016','01')


# In[ ]:


ds_ori_2008 = list_ds(2008,scen1,'off')
ds_ori_2016 = list_ds(2016,scen1,'off')


# In[ ]:


ds_AIRS_AM_08 = ds_ori_2008['ds_AIRS_AM']
ds_AIRS_PM_08 = ds_ori_2008['ds_AIRS_PM']
ds_IASI_AM_08 = ds_ori_2008['ds_IASI_AM']
ds_IASI_PM_08 = ds_ori_2008['ds_IASI_PM']

ds_AIRS_AM_16 = ds_ori_2016['ds_AIRS_AM']
ds_AIRS_PM_16 = ds_ori_2016['ds_AIRS_PM']
ds_IASI_AM_16 = ds_ori_2016['ds_IASI_AM']
ds_IASI_PM_16 = ds_ori_2016['ds_IASI_PM']




#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Profiles !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# heating profiles cutted into 3 cases (no rain, light rain and heavy rain)


# In[ ]:


heat_profile_3cases(ds_hgh_08_3cases['ds_LH_all'],ds_hgh_16_3cases['ds_LH_all'],'LH',scen1)


# In[ ]:


heat_profile_3cases(ds_mlow_08_3cases['ds_LH_all'],ds_mlow_16_3cases['ds_LH_all'],'LH',scen2)


# In[ ]:


heat_profile_3cases(ds_hgh_08_3cases['ds_LW_all'],ds_hgh_16_3cases['ds_LW_all'],'LW',scen1)


# In[ ]:


heat_profile_3cases(ds_mlow_08_3cases['ds_LW_all'],ds_mlow_16_3cases['ds_LW_all'],'LW',scen2)


# In[ ]:


heat_profile_3cases(ds_hgh_08_3cases['ds_SW_all'],ds_hgh_16_3cases['ds_SW_all'],'SW',scen1)


# In[ ]:


heat_profile_3cases(ds_mlow_08_3cases['ds_SW_all'],ds_mlow_16_3cases['ds_SW_all'],'SW',scen2)



# LH, LH + CRE profiles for different scenes


# In[ ]:


LH_CRE_profile(ds_all_08['ds_LH_AIRS_AM'],ds_all_08['ds_LH_AIRS_PM'],
               ds_all_08['ds_LW_AIRS_AM'],ds_all_08['ds_LW_AIRS_PM'],
               ds_all_08['ds_SW_AIRS_AM'],ds_all_08['ds_SW_AIRS_PM'],
               clr_info_08['clrLW_AIRS_AM'],clr_info_08['clrLW_AIRS_PM'],clr_info_08['clrSW_AIRS_PM'],
               'AIRS',2008,scen_all)


# In[ ]:


LH_CRE_profile(ds_clr_08['ds_LH_AIRS_AM'],ds_clr_08['ds_LH_AIRS_PM'],
               ds_clr_08['ds_LW_AIRS_AM'],ds_clr_08['ds_LW_AIRS_PM'],
               ds_clr_08['ds_SW_AIRS_AM'],ds_clr_08['ds_SW_AIRS_PM'],
               clr_info_08['clrLW_AIRS_AM'],clr_info_08['clrLW_AIRS_PM'],clr_info_08['clrSW_AIRS_PM'],
               'AIRS',2008,scen0)


# In[ ]:


LH_CRE_profile(ds_hgh_08['ds_LH_AIRS_AM'],ds_hgh_08['ds_LH_AIRS_PM'],
               ds_hgh_08['ds_LW_AIRS_AM'],ds_hgh_08['ds_LW_AIRS_PM'],
               ds_hgh_08['ds_SW_AIRS_AM'],ds_hgh_08['ds_SW_AIRS_PM'],
               clr_info_08['clrLW_AIRS_AM'],clr_info_08['clrLW_AIRS_PM'],clr_info_08['clrSW_AIRS_PM'],
               'AIRS',2008,scen1)


# In[ ]:


LH_CRE_profile(ds_mlow_08['ds_LH_AIRS_AM'],ds_mlow_08['ds_LH_AIRS_PM'],
               ds_mlow_08['ds_LW_AIRS_AM'],ds_mlow_08['ds_LW_AIRS_PM'],
               ds_mlow_08['ds_SW_AIRS_AM'],ds_mlow_08['ds_SW_AIRS_PM'],
               clr_info_08['clrLW_AIRS_AM'],clr_info_08['clrLW_AIRS_PM'],clr_info_08['clrSW_AIRS_PM'],
               'AIRS',2008,scen2)


# In[ ]:


LH_CRE_profile(ds_all_16['ds_LH_AIRS_AM'],ds_all_16['ds_LH_AIRS_PM'],
               ds_all_16['ds_LW_AIRS_AM'],ds_all_16['ds_LW_AIRS_PM'],
               ds_all_16['ds_SW_AIRS_AM'],ds_all_16['ds_SW_AIRS_PM'],
               clr_info_16['clrLW_AIRS_AM'],clr_info_16['clrLW_AIRS_PM'],clr_info_16['clrSW_AIRS_PM'],
               'AIRS',2016,scen_all)


# In[ ]:


LH_CRE_profile(ds_hgh_16['ds_LH_AIRS_AM'],ds_hgh_16['ds_LH_AIRS_PM'],
               ds_hgh_16['ds_LW_AIRS_AM'],ds_hgh_16['ds_LW_AIRS_PM'],
               ds_hgh_16['ds_SW_AIRS_AM'],ds_hgh_16['ds_SW_AIRS_PM'],
               clr_info_16['clrLW_AIRS_AM'],clr_info_16['clrLW_AIRS_PM'],clr_info_16['clrSW_AIRS_PM'],
               'AIRS',2016,scen0)


# In[ ]:


LH_CRE_profile(ds_hgh_16['ds_LH_AIRS_AM'],ds_hgh_16['ds_LH_AIRS_PM'],
               ds_hgh_16['ds_LW_AIRS_AM'],ds_hgh_16['ds_LW_AIRS_PM'],
               ds_hgh_16['ds_SW_AIRS_AM'],ds_hgh_16['ds_SW_AIRS_PM'],
               clr_info_16['clrLW_AIRS_AM'],clr_info_16['clrLW_AIRS_PM'],clr_info_16['clrSW_AIRS_PM'],
               'AIRS',2016,scen1)


# In[ ]:


LH_CRE_profile(ds_hgh_16['ds_LH_AIRS_AM'],ds_hgh_16['ds_LH_AIRS_PM'],
               ds_hgh_16['ds_LW_AIRS_AM'],ds_hgh_16['ds_LW_AIRS_PM'],
               ds_hgh_16['ds_SW_AIRS_AM'],ds_hgh_16['ds_SW_AIRS_PM'],
               clr_info_16['clrLW_AIRS_AM'],clr_info_16['clrLW_AIRS_PM'],clr_info_16['clrSW_AIRS_PM'],
               'AIRS',2016,scen2)



# LH, LH + CRE profiles for different scenes. 2008 vs 2016 


# In[ ]:


LH_CRE_profile_0816(
    ds_hgh_08['ds_LH_AIRS_AM'],ds_hgh_08['ds_LH_AIRS_PM'],ds_hgh_08['ds_LW_AIRS_AM'],ds_hgh_08['ds_LW_AIRS_PM'],ds_hgh_08['ds_SW_AIRS_AM'],ds_hgh_08['ds_SW_AIRS_PM'],
    ds_hgh_16['ds_LH_AIRS_AM'],ds_hgh_16['ds_LH_AIRS_PM'],ds_hgh_16['ds_LW_AIRS_AM'],ds_hgh_16['ds_LW_AIRS_PM'],ds_hgh_16['ds_SW_AIRS_AM'],ds_hgh_16['ds_SW_AIRS_PM'],
    clr_info_16['clrLW_AIRS_AM'],clr_info_16['clrLW_AIRS_PM'],clr_info_16['clrSW_AIRS_PM'],
    'AIRS',
    scen1
)


# In[ ]:


LH_CRE_profile_0816(
    ds_mlow_08['ds_LH_AIRS_AM'],ds_mlow_08['ds_LH_AIRS_PM'],ds_mlow_08['ds_LW_AIRS_AM'],ds_mlow_08['ds_LW_AIRS_PM'],ds_mlow_08['ds_SW_AIRS_AM'],ds_mlow_08['ds_SW_AIRS_PM'],
    ds_mlow_16['ds_LH_AIRS_AM'],ds_mlow_16['ds_LH_AIRS_PM'],ds_mlow_16['ds_LW_AIRS_AM'],ds_mlow_16['ds_LW_AIRS_PM'],ds_mlow_16['ds_SW_AIRS_AM'],ds_mlow_16['ds_SW_AIRS_PM'],
    clr_info_16['clrLW_AIRS_AM'],clr_info_16['clrLW_AIRS_PM'],clr_info_16['clrSW_AIRS_PM'],
    'AIRS',
    scen2
)


# In[ ]:


LH_CRE_profile_0816(
    ds_all_08['ds_LH_AIRS_AM'],ds_all_08['ds_LH_AIRS_PM'],ds_all_08['ds_LW_AIRS_AM'],ds_all_08['ds_LW_AIRS_PM'],ds_all_08['ds_SW_AIRS_AM'],ds_all_08['ds_SW_AIRS_PM'],
    ds_all_16['ds_LH_AIRS_AM'],ds_all_16['ds_LH_AIRS_PM'],ds_all_16['ds_LW_AIRS_AM'],ds_all_16['ds_LW_AIRS_PM'],ds_all_16['ds_SW_AIRS_AM'],ds_all_16['ds_SW_AIRS_PM'],
    clr_info_16['clrLW_AIRS_AM'],clr_info_16['clrLW_AIRS_PM'],clr_info_16['clrSW_AIRS_PM'],
    'AIRS',
    scen_all
)



#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Maps !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# In[ ]:


LH_map_0816(ds_AIRS_AM_08,ds_AIRS_PM_08,200,550,'AIRS','day',2008)


# In[ ]:


LH_map_0816(ds_AIRS_AM_16,ds_AIRS_PM_16,200,550,'AIRS','day',2016)
 
    

# CRE maps (am, pm, am+pm respectively)




CRE_map_0816(
    ds_AIRS_AM_08,ds_AIRS_PM_08,
    clr_info_08['clrLW_AIRS_AM_map'],clr_info_08['clrLW_AIRS_PM_map'],clr_info_08['clrSW_AIRS_PM_map'],
    200,550,
    'AIRS',
    'day',
    '2008'
) 


# In[ ]:


CRE_map_0816(
    ds_AIRS_AM_08,ds_AIRS_PM_08,
    clr_info_08['clrLW_AIRS_AM_map'],clr_info_08['clrLW_AIRS_PM_map'],clr_info_08['clrSW_AIRS_PM_map'],
    200,550,
    'AIRS',
    'AM',
    '2008'
)


# In[ ]:


CRE_map_0816(
    ds_AIRS_AM_08,ds_AIRS_PM_08,
    clr_info_08['clrLW_AIRS_AM_map'],clr_info_08['clrLW_AIRS_PM_map'],clr_info_08['clrSW_AIRS_PM_map'],
    200,550,
    'AIRS',
    'PM',
    '2008'
)



# In[ ]:


CRE_map_0816(
    ds_AIRS_AM_16,ds_AIRS_PM_16,
    clr_info_16['clrLW_AIRS_AM_map'],clr_info_16['clrLW_AIRS_PM_map'],clr_info_16['clrSW_AIRS_PM_map'],
    200,550,
    'AIRS',
    'day',
    '2016'
) 


# In[ ]:


CRE_map_0816(
    ds_AIRS_AM_16,ds_AIRS_PM_16,
    clr_info_16['clrLW_AIRS_AM_map'],clr_info_16['clrLW_AIRS_PM_map'],clr_info_16['clrSW_AIRS_PM_map'],
    200,550,
    'AIRS',
    'AM',
    '2016'
) 


# In[ ]:


CRE_map_0816(
    ds_AIRS_AM_16,ds_AIRS_PM_16,
    clr_info_16['clrLW_AIRS_AM_map'],clr_info_16['clrLW_AIRS_PM_map'],clr_info_16['clrSW_AIRS_PM_map'],
    200,550,
    'AIRS',
    'PM',
    '2016'
) 





# LH El nino-La nina difference maps (2016 - 2008)  


# In[ ]:


LH_map_diff_0816(ds_AIRS_AM_08,ds_AIRS_PM_08,ds_AIRS_AM_16,ds_AIRS_PM_16,100,200,'AIRS','day','0816')


# In[ ]:


LH_map_diff_0816(ds_AIRS_AM_08,ds_AIRS_PM_08,ds_AIRS_AM_16,ds_AIRS_PM_16,200,550,'AIRS','day','0816')


# In[ ]:


LH_map_diff_0816(ds_AIRS_AM_08,ds_AIRS_PM_08,ds_AIRS_AM_16,ds_AIRS_PM_16,550,900,'AIRS','day','0816')




# CRE El nino-La nina difference maps (2016 - 2008) for different pressure levels. 

CRE_map_diff_0816(
    ds_AIRS_AM_08,ds_AIRS_PM_08,ds_AIRS_AM_16,ds_AIRS_PM_16,
    clr_info_08['clrLW_AIRS_AM_map'],clr_info_08['clrLW_AIRS_PM_map'],clr_info_08['clrSW_AIRS_PM_map'],
    clr_info_16['clrLW_AIRS_AM_map'],clr_info_16['clrLW_AIRS_PM_map'],clr_info_16['clrSW_AIRS_PM_map'],
    100,200,
    'AIRS',
    'day',
    '0816'
)  


# In[ ]:


CRE_map_diff_0816(
    ds_AIRS_AM_08,ds_AIRS_PM_08,ds_AIRS_AM_16,ds_AIRS_PM_16,
    clr_info_08['clrLW_AIRS_AM_map'],clr_info_08['clrLW_AIRS_PM_map'],clr_info_08['clrSW_AIRS_PM_map'],
    clr_info_16['clrLW_AIRS_AM_map'],clr_info_16['clrLW_AIRS_PM_map'],clr_info_16['clrSW_AIRS_PM_map'],
    200,550,
    'AIRS',
    'day',
    '0816'
)


# In[ ]:


CRE_map_diff_0816(
    ds_AIRS_AM_08,ds_AIRS_PM_08,ds_AIRS_AM_16,ds_AIRS_PM_16,
    clr_info_08['clrLW_AIRS_AM_map'],clr_info_08['clrLW_AIRS_PM_map'],clr_info_08['clrSW_AIRS_PM_map'],
    clr_info_16['clrLW_AIRS_AM_map'],clr_info_16['clrLW_AIRS_PM_map'],clr_info_16['clrSW_AIRS_PM_map'],
    550,900,
    'AIRS',
    'day',
    '0816'
)



# Clear sky CRE maps 


# In[ ]:


CRE_map_0816_clrsky(
    clr_info_08['clrLW_AIRS_AM_map'],clr_info_08['clrLW_AIRS_PM_map'],clr_info_08['clrSW_AIRS_PM_map'],
    200,550,
    'AIRS',
    'day',
    '2008'
) 

# after filled by function "interpolate_na()"


# In[ ]:


CRE_map_0816_clrsky(
    clr_info_16['clrLW_AIRS_AM_map'],clr_info_16['clrLW_AIRS_PM_map'],clr_info_16['clrSW_AIRS_PM_map'],
    200,550,
    'AIRS',
    'AM',
    '2016'
) 


# In[ ]:


CRE_map_0816_clrsky(
    clr_info_16['clrLW_AIRS_AM_map'],clr_info_16['clrLW_AIRS_PM_map'],clr_info_16['clrSW_AIRS_PM_map'],
    200,550,
    'AIRS',
    'PM',
    '2016'
) 



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Cloud systems !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# In[ ]:


# needs to change to read all data of several years afterwards !!!

# Read clouds systems .nc data
ds_sys_AM_08 = xr.open_dataset('/bdd/ARA/GEWEX_CA_ftp/cloud_systems/HLC90_DP8lnp_P350_CIRSv2/AIRS_0.5_dd/phase2_Cb93.98/clustersP350_AIRS_0.5_dd_08_01_0130AM_CIRSv2.nc')
ds_sys_PM_08 = xr.open_dataset('/bdd/ARA/GEWEX_CA_ftp/cloud_systems/HLC90_DP8lnp_P350_CIRSv2/AIRS_0.5_dd/phase2_Cb93.98/clustersP350_AIRS_0.5_dd_08_01_0130PM_CIRSv2.nc')
ds_sys_AM_16 = xr.open_dataset('/bdd/ARA/GEWEX_CA_ftp/cloud_systems/HLC90_DP8lnp_P350_CIRSv2/AIRS_0.5_dd/phase2_Cb93.98/clustersP350_AIRS_0.5_dd_16_01_0130AM_CIRSv2.nc')
ds_sys_PM_16 = xr.open_dataset('/bdd/ARA/GEWEX_CA_ftp/cloud_systems/HLC90_DP8lnp_P350_CIRSv2/AIRS_0.5_dd/phase2_Cb93.98/clustersP350_AIRS_0.5_dd_16_01_0130PM_CIRSv2.nc')

def ds_cld_sys(ds_sys):
    
    #ds_sys = ds_sys.mean('time')
    
    # slice lon and lat to make sure that data has same dim as LH
    lon_slice = slice(-180, 179.5)  
    lat_slice = slice(-30, 29.5)  
    ds_sys = ds_sys.sel(lon=lon_slice, lat=lat_slice)

    return(ds_sys)

ds_sys_AM_08 = ds_cld_sys(ds_sys_AM_08)
ds_sys_PM_08 = ds_cld_sys(ds_sys_PM_08)
ds_sys_AM_16 = ds_cld_sys(ds_sys_AM_16)
ds_sys_PM_16 = ds_cld_sys(ds_sys_PM_16)


# In[ ]:


LH_sys_map_0816(ds_AIRS_AM_16,ds_AIRS_PM_16,ds_sys_AM_16,ds_sys_AM_16,0,200,550,'AIRS','AM','2016')


# In[ ]:


LH_sys_map_0816(ds_AIRS_AM_16,ds_AIRS_PM_16,ds_sys_AM_16,ds_sys_AM_16,0,200,550,'AIRS','PM','2016')


# In[ ]:


print(f'This script needed {(datetime.datetime.now() - start_time).seconds} seconds') 

