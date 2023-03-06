#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

import xarray as xr
import os
import calendar
import datetime

import warnings
warnings.filterwarnings("ignore")

start_time = datetime.datetime.now()


# In[3]:


"""
Write LH,HR,Vertical structure data into one single .nc file (daily file)
"""

year = 2008
ampm ='PM'
sat = 'AIRS'
time = '0130' # 0130 for AIRS, 0930 for IASI


RV_path = f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/results_phase2_data-day/{year}/'  #loholt 
LH_path = f'/bdd/ARA/GEWEX_CA_ftp/deep_learning/RRind3_LH_nan0_phase2_data-day/{year}/'

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
            ds_all = ds_Vert.merge(ds_LH,compat='override').merge(ds_LW,compat='override').merge(ds_SW,compat='override')
            
        else:
            ds_all = ds_Vert.merge(ds_LH,compat='override').merge(ds_LW,compat='override')
                   
        
        ds_all = ds_all.drop('latitude').drop('longitude')
        
        outfilename = f'LH_HR_Vert_{sat}_{time}{ampm}_{year}{month:02d}{day:02d}.nc'
        out_path = f'/homedata/xchen/combined_LH_HR_Vert/{year}/{month:02d}/' 
        outfile = os.path.join(out_path, outfilename)
        
        ds_all.to_netcdf(outfile)
        
        #ds_all.to_netcdf(to_path + f'LH_HR_Vert_{sat}_{time}{ampm}_{year}{month:02d}{day:02d}.nc')
        
        
print(f'This script needed {(datetime.datetime.now() - start_time).seconds} seconds')                                


# In[ ]:




