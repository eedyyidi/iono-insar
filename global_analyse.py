import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import h5py
import numpy as np
import os
import datetime as dt
import numpy as np
import math
from scipy import interpolate
from mintpy.objects import ionex, timeseries
from mintpy.simulation import iono
from mintpy.utils import readfile, writefile
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from mintpy import iono_tec
from mintpy.cli import diff, ifgram_inversion, modify_network, reference_point, reference_date
from mintpy.utils import utils as ut
from tqdm import trange
import h5py

from datetime import datetime
from netCDF4 import Dataset

tframe_left = gpd.read_file("/home/eedy/data/aux/tframe_orbit/tframe_left_look.gpkg")
tframe_left.crs = "EPSG:4326"

# 初始化列表
ascending_frame_time_date = []
ascending_frame_time_utc_sec = []
ascending_NearLookAngle = []
ascending_FarLookAngle = []
ascending_lat1 = []
ascending_lon1 = []
ascending_lat2 = []
ascending_lon2 = []
ascending_lat3 = []
ascending_lon3 = []
ascending_lat4 = []
ascending_lon4 = []

descending_frame_time_date = []
descending_frame_time_utc_sec = []
descending_latitude = []
descending_longitude = []
descending_NearLookAngle = []
descending_FarLookAngle = []
descending_lat1 = []
descending_lon1 = []
descending_lat2 = []
descending_lon2 = []
descending_lat3 = []
descending_lon3 = []
descending_lat4 = []
descending_lon4 = []

# 遍历 track_frame
for index, row in tframe_left.iterrows():
    # 获取日期和时间
    startET = row['startET']
    endET = row['endET']
    tcenter = (startET + endET)/2.0
    import datetime
    frame_time_utc = datetime.datetime(2000,1,1)+datetime.timedelta(seconds=tcenter)
    frame_time_date = frame_time_utc.date()
    frame_time_utc_sec = frame_time_utc.hour * 60 + frame_time_utc.minute + frame_time_utc.second / 60

    # 获取入射角
    swathNearLookAngle = row['swathNearLookAngle']
    swathFarLookAngle = row['swathFarLookAngle']

    polygon = row['geometry']
    exterior_coords = polygon.geoms[0].exterior.coords
    # 提取四个角点
    corners = [exterior_coords[0], exterior_coords[len(exterior_coords) // 4],exterior_coords[len(exterior_coords) // 2], exterior_coords[3 * len(exterior_coords) // 4]]
    ## 1:右下  2:左下  3:左上  4:右上
    ## 1\4:FarLookAngle  2\3:NearLookAngle

    lat1 = corners[0][1]
    lon1 = corners[0][0]
    lat2 = corners[1][1]
    lon2 = corners[1][0]
    lat3 = corners[2][1]
    lon3 = corners[2][0]
    lat4 = corners[3][1]
    lon4 = corners[3][0]
    
    
    # 根据 passDirection 来决定是 ascending 还是 descending
    pass_direction = row['passDirection']
    
    if pass_direction == 'Ascending':
        ascending_frame_time_date.append(frame_time_date)
        ascending_frame_time_utc_sec.append(frame_time_utc_sec)
        ascending_NearLookAngle.append(swathNearLookAngle)
        ascending_FarLookAngle.append(swathFarLookAngle)
        ascending_lat1.append(lat1)
        ascending_lon1.append(lon1)
        ascending_lat2.append(lat2)
        ascending_lon2.append(lon2)
        ascending_lat3.append(lat3)
        ascending_lon3.append(lon3)
        ascending_lat4.append(lat4)
        ascending_lon4.append(lon4)
        
    elif pass_direction == 'Descending':
        descending_frame_time_date.append(frame_time_date)
        descending_frame_time_utc_sec.append(frame_time_utc_sec)
        descending_NearLookAngle.append(swathNearLookAngle)
        descending_FarLookAngle.append(swathFarLookAngle)
        descending_lat1.append(lat1)
        descending_lon1.append(lon1)
        descending_lat2.append(lat2)
        descending_lon2.append(lon2)
        descending_lat3.append(lat3)
        descending_lon3.append(lon3)
        descending_lat4.append(lat4)
        descending_lon4.append(lon4)
    else:
        # 处理其他情况，如果有的话
        pass

def compute_lat_lon_ipp(latitude , longitude , LookAngle , frame_time_utc_sec , azimuthAngle = -256):
    incidenceAngle = LookAngle

    azimuthAngle = azimuthAngle                 ########################假设升轨的方位角-256,降轨的方位角-104

    latitude = latitude

    longitude = longitude
    
    theta = incidenceAngle*np.pi/180
    Re = 6371000
    h_ipp = 450e3
    theta_ipp = np.arcsin(Re*np.sin(theta)/(Re+h_ipp))
    HEADING = azimuthAngle*np.pi/180

    alpha_ipp = theta - theta_ipp

    latitude_pi = latitude *np.pi/180
    longitude_pi = longitude *np.pi/180

    lat_ipp_pi = np.arcsin(np.sin(latitude_pi)*np.cos(alpha_ipp) + np.cos(latitude_pi)*np.sin(alpha_ipp)*np.cos(HEADING))
    atan2_func = np.vectorize(math.atan2)
    delta = atan2_func(-np.sin(alpha_ipp)*np.cos(latitude_pi)*np.sin(HEADING), np.cos(alpha_ipp) - np.sin(latitude_pi)*np.sin(lat_ipp_pi))
    lon_ipp_pi = np.mod(longitude_pi + delta + np.pi, 2*np.pi) - np.pi

    lat_ipp = lat_ipp_pi *180/np.pi
    lon_ipp = lon_ipp_pi *180/np.pi

    return lat_ipp, lon_ipp ,theta_ipp

def read_netcdf_file(frame_time_date):
    # Open the NetCDF file for reading
    date_obj = frame_time_date
    date_str = date_obj.strftime('%Y-%m-%d')
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    day_of_year = date_obj.timetuple().tm_yday
    year_last_two = date_obj.strftime('%y')
    filename = f"/home/eedy/data/aux/IONEX/jpld{day_of_year:03d}0.{year_last_two}i.nc"

    dataset = Dataset(filename, 'r')

    # Get the data variables
    varepochs_data = dataset.variables['varepochs'][:]
    time_data = dataset.variables['time'][:]
    lats = dataset.variables['lat'][:]
    lons = dataset.variables['lon'][:]
    tec_maps = dataset.variables['tecmap'][:]
    tec_flag = dataset.variables['tecflag'][:]

    # Close the NetCDF file
    dataset.close()

    mins = []
    for varepoch in varepochs_data:
        dt = datetime.strptime(varepoch, "%Y/%m/%d_%H:%M:%S")
        total_minutes = dt.hour * 60 + dt.minute + dt.second / 60
        mins.append(total_minutes)

    # 将 MaskedArray 转换为普通的 Python 列表
    lons = lons.tolist()
    # 要添加的一系列值（带有小数部分）
    additional_lons_before = [x - 0.5 for x in range(-200, -179)]
    # 将这些值依次添加到 lons 列表的前面
    lons = additional_lons_before + lons
    # 要添加的一系列值（带有小数部分）
    additional_lons = [x + 0.5 for x in range(180, 201)]
    # 将这些值依次添加到 lons 列表中
    lons.extend(additional_lons)

    # 假设 tec_maps 的形状为 (96, 180, 360)
    tec_maps_shape = tec_maps.shape
    # 创建一个与 tec_maps 形状相同的零数组，形状为 (96, 180, 381)
    expanded_tec_maps = np.zeros((tec_maps_shape[0], tec_maps_shape[1], tec_maps_shape[2] + 42))
    # 对 tec_maps 进行扩展
    for i in range(tec_maps_shape[0]):
        expanded_tec_maps[i, :, :] = np.concatenate((tec_maps[i, :, 338:359], tec_maps[i, :, :],  tec_maps[i, :, :21]), axis=1)
    tec_maps = expanded_tec_maps
    
    # Return the data
    return varepochs_data, time_data, lats, lons, tec_maps, tec_flag , mins

def compute_r_iono(frame_time_utc_sec, frame_time_date, latitude , longitude, LookAngle , azimuthAngle = -256):
    lat_ipp, lon_ipp ,theta_ipp = compute_lat_lon_ipp(latitude , longitude, LookAngle , frame_time_utc_sec , azimuthAngle = -256 )

    if lon_ipp >=  180 :
        lon_ipp -= 360.
    if lon_ipp <=  -180 :
        lon_ipp += 360.
    
    varepochs, time, lats, lons, tec_maps, tecflag , mins= read_netcdf_file(frame_time_date)
    
    minutes1 = frame_time_utc_sec
    for i in range(len(mins) - 1):
        if minutes1 >= mins[i] and minutes1 <= mins[i+1]:
            break

    valid_mask = ~(np.isnan(lat_ipp) | np.isnan(lon_ipp))
    valid_lat_ipp = lat_ipp[valid_mask]
    valid_lon_ipp = lon_ipp[valid_mask]

    interp_func = RegularGridInterpolator((mins, lats, lons), tec_maps, method='linear')

    Ei = np.column_stack((np.full(valid_lat_ipp.size, mins[i]),
                          valid_lat_ipp,
                          valid_lon_ipp + (minutes1 - mins[i]) * 360. / (24. * 60.)))
    
    Ei1 = np.column_stack((np.full(valid_lat_ipp.size, mins[i+1]),
                           valid_lat_ipp,
                           valid_lon_ipp + (minutes1 - mins[i+1]) * 360. / (24. * 60.)))

    new_tec_map1 = np.full_like(lat_ipp, np.nan)
    new_tec_map1[valid_mask] = ((mins[i+1] - minutes1) / (mins[i+1] - mins[i]) * interp_func(Ei) + (minutes1 - mins[i]) / (mins[i+1] - mins[i]) * interp_func(Ei1))

    k = 40.31
    c = 299792458
    freq = 1.257e9
    h_ipp = 450e3
    Re = 6371000
   
    VTEC = new_tec_map1*1e16
    a = VTEC * k / (freq ** 2)
    r_iono = a / np.cos(np.arcsin(np.sin(theta_ipp) / (1 + a)))
    
    return r_iono


import datetime
start_date = datetime.datetime(2022, 1, 1)
end_date = datetime.datetime(2022, 12, 31)
delta = datetime.timedelta(days=1)

time_date = []

current_date = start_date
while current_date <= end_date:
    time_date.append(current_date.date())
    current_date += delta


for k in trange(len(time_date), desc='Processing'):
    r_iono1 = []
    r_iono2 = []
    r_iono3 = []
    r_iono4 = []
    
    for i in trange(len(ascending_frame_time_utc_sec), desc='Inner Processing'):
        try:
            from datetime import datetime
            result = compute_r_iono(ascending_frame_time_utc_sec[i], time_date[k], ascending_lat1[i], ascending_lon1[i], ascending_FarLookAngle[i], azimuthAngle=-256)
        except Exception as e:
            print(f"Error occurred: {e}")
            result = np.nan
        r_iono1.append(result)

        try:
            from datetime import datetime
            result = compute_r_iono(ascending_frame_time_utc_sec[i], time_date[k], ascending_lat2[i], ascending_lon2[i], ascending_NearLookAngle[i], azimuthAngle=-256)
        except Exception as e:
            print(f"Error occurred: {e}")
            result = np.nan
        r_iono2.append(result)

        try:
            from datetime import datetime
            result = compute_r_iono(ascending_frame_time_utc_sec[i], time_date[k], ascending_lat3[i], ascending_lon3[i], ascending_NearLookAngle[i], azimuthAngle=-256)
        except Exception as e:
            print(f"Error occurred: {e}")
            result = np.nan
        r_iono3.append(result)

        try:
            from datetime import datetime
            result = compute_r_iono(ascending_frame_time_utc_sec[i], time_date[k], ascending_lat4[i], ascending_lon4[i], ascending_FarLookAngle[i], azimuthAngle=-256)
        except Exception as e:
            print(f"Error occurred: {e}")
            result = np.nan
        r_iono4.append(result)
    
    # 打开一个新的.h5文件，如果文件已存在，则覆盖
    with h5py.File(f'data/global/asc_iono_{k}.h5', 'w') as h5f:
        # 将数据写入.h5文件
        h5f.create_dataset('r_iono1', data=r_iono1)
        h5f.create_dataset('r_iono2', data=r_iono2)
        h5f.create_dataset('r_iono3', data=r_iono3)
        h5f.create_dataset('r_iono4', data=r_iono4)









