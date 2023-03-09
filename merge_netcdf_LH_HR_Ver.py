#!/usr/bin/env python
# coding: utf-8

"""
Created on Thu Mar  9 16:52:42 2023

@author: Xiaoting CHEN
"""

import xarray as xr
import os
import calendar
import datetime

import warnings
warnings.filterwarnings("ignore")

start_time = datetime.datetime.now()


"""
Write LH,HR,Vertical structure data into one single .nc file (daily file)
"""

year = 2008
ampm ='PM'
sat = 'AIRS'
time = '0130' # 0130 for AIRS, 0930 for IASI
model_type = '_'  # '_' for with ML, '_noML_' for without ML


RV_path = f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/'  #loholt 
LH_path = f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/RRind3{model_type}LH_nan0_phase2_data-day/{year}/'

HR_level = np.array([ 78.,  96., 119., 147., 181., 211., 
                     235., 262., 292., 325., 361., 402., 
                     448., 498., 555., 618., 688., 762., 
                     824., 875., 928., 984.]
                    )

#################################### Merge LH, LW, SW and VertRR data to one single dataset ####################################

for month in range(1,12+1): 
    
    days_in_month = calendar.monthrange(year, month)[1]    
    
    path_RV = RV_path + f'{month:02d}/'
    path_LH = LH_path + f'{month:02d}/'
    
    for day in range(1, days_in_month + 1):
        
        #print(f"{year}-{month:02d}-{day:02d}")
        ds_Vert = xr.open_dataset(path_RV+f'L3_VertRR_CIRS-{sat}v2_{time}{ampm}_{year}{month:02d}{day:02d}.nc')
        
        ds_LH = xr.open_dataset(path_LH+f'LH_{sat}_{year}{month:02d}{day:02d}_{time}{ampm}.nc')
        ds_LH = ds_LH.swap_dims({'latitude':'lat','longitude':'lon'})
        
        ds_LW = xr.open_dataset(path_RV+f'LW_{sat}-ERAI_v8_{time}{ampm}_{year}{month:02d}{day:02d}.nc')
        
        if (time == '0130' and ampm =='PM') or (time == '0930' and ampm =='AM'):
            
            ds_SW = xr.open_dataset(path_RV+f'SW_{sat}-ERAI_v8_0130{ampm}_{year}{month:02d}{day:02d}.nc')
            ds_all = xr.merge([ds_Vert,ds_LH, ds_LW, ds_SW],compat='override')
            
        else:
            ds_all = xr.merge([ds_Vert,ds_LH, ds_LW],compat='override')
                   
        
        ds_all = ds_all.drop(['latitude', 'longitude'])
        
################################### Concat 22 LW and SW variables to a new 3D variable #######################################       
        
        # Variable name list to merge
        LW_names = []
        SW_names = []
        
        for i in range(22):
            var_LW = f'LW{i}'
            var_SW = f'SW{i}'
            
            LW_names.append(var_LW)
            SW_names.append(var_SW)
            
        
        # Merge variables into a new variable LW, SW
        ds_all['LW'] = xr.concat([ds_all[LW_name] for LW_name in  LW_names], dim='HR_level')
        ds_all = ds_all.drop_vars(LW_names)
        
        #SW just exist for day-time, check it exists here or not!
        if SW_names[0] in ds_all.data_vars:
            ds_all['SW'] = xr.concat([ds_all[SW_name] for SW_name in  SW_names], dim='HR_level')
            ds_all = ds_all.drop_vars(SW_names)
            
        #Fill in the new dimension：HR_level (do not drop the first 2 layers, cuz HR is a little bit higher than LH)
        ds_all.update({'HR_level': HR_level})      
        
        
#################################################### Fill attributes information #############################################
        ds_all.attrs.update({'Satellite' : f'{sat}',
                             'Institution': 'LMD/IPSL',
                             'Grid_resolution': '0.5 x 0.5 deg',
                             'Author': 'CHEN Xiaoting'
                            })
    
######################################################### Creat new files.nc #################################################
        outfilename = f'LH_HR_Vert_{sat}_{time}{ampm}_{year}{month:02d}{day:02d}.nc'
        out_path = f'/homedata/xchen/combined{model_type}LH_HR_Vert/{year}/{month:02d}/' 
        outfile = os.path.join(out_path, outfilename)
        
        ds_all.to_netcdf(outfile)
        
        
print(f'This script needed {(datetime.datetime.now() - start_time).seconds} seconds') 
