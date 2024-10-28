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
from mintpy.utils import ptime, readfile, writefile

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

tec_dir = '/home/eedy/data/aux/IONEX'
tec_files = []
year = 2010
date_list = ptime.get_date_range(f'{year}0101', f'{year}1231')
tec_files = iono_tec.download_ionex_files(date_list, tec_dir, sol_code='jpl')

tec_maps = []
for tec_file in tec_files:
    try:
        mins, lats, lons, tec_map, rms_maps = ionex.read_ionex(tec_file)
        tec_maps.append(tec_map)
    except Exception as e:
        print(f"Error occurred: {e}")
        tec_map = np.nan


tec_maps_array = np.array(tec_maps)

lons = lons.tolist()
    # 要添加的一系列值（带有小数部分）
additional_lons_before = [-230,-225,-220,-215,-210,-205,-200,-195,-190,-185]
    # 将这些值依次添加到 lons 列表的前面
lons = additional_lons_before + lons
    # 要添加的一系列值（带有小数部分）
additional_lons = [185,190,195,200,205,210,215,220,225,230]
    # 将这些值依次添加到 lons 列表中
lons.extend(additional_lons)

tec_maps_shape = tec_maps_array.shape
    # 创建一个与 tec_maps 形状相同的零数组，形状为 (96, 180, 381)
expanded_tec_maps = np.zeros((tec_maps_shape[0], tec_maps_shape[1], tec_maps_shape[2],tec_maps_shape[3] + 20))
    # 对 tec_maps 进行扩展
for i in range(tec_maps_shape[0]):
    expanded_tec_maps[i] = np.concatenate((tec_maps_array[i, :, :, 62:72], tec_maps_array[i], tec_maps_array[i, :, :, :10]), axis=2)
tec_maps = expanded_tec_maps
tec_maps_array = np.array(tec_maps)

def compute_lat_lon_ipp(latitude , longitude , LookAngle ,  azimuthAngle = -256):
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

    return lat_ipp, lon_ipp 

lat_ipp_1 = []
lon_ipp_1 = []
for i in trange(len(ascending_lat1), desc='Inner Processing'):
    lat_ipp_i , lon_ipp_i = compute_lat_lon_ipp(ascending_lat1[i] , ascending_lon1[i] , ascending_NearLookAngle[i] ,  azimuthAngle = -256)
    lat_ipp_1.append(lat_ipp_i)
    lon_ipp_1.append(lon_ipp_i)

lat_ipp_2 = []
lon_ipp_2 = []
for i in trange(len(ascending_lat1), desc='Inner Processing'):
    lat_ipp_i , lon_ipp_i = compute_lat_lon_ipp(ascending_lat2[i] , ascending_lon2[i] , ascending_FarLookAngle[i] ,  azimuthAngle = -256)
    lat_ipp_2.append(lat_ipp_i)
    lon_ipp_2.append(lon_ipp_i)

lat_ipp_3 = []
lon_ipp_3 = []
for i in trange(len(ascending_lat1), desc='Inner Processing'):
    lat_ipp_i , lon_ipp_i = compute_lat_lon_ipp(ascending_lat3[i] , ascending_lon3[i] , ascending_FarLookAngle[i] ,  azimuthAngle = -256)
    lat_ipp_3.append(lat_ipp_i)
    lon_ipp_3.append(lon_ipp_i)

lat_ipp_4 = []
lon_ipp_4 = []
for i in trange(len(ascending_lat1), desc='Inner Processing'):
    lat_ipp_i , lon_ipp_i = compute_lat_lon_ipp(ascending_lat4[i] , ascending_lon4[i] , ascending_NearLookAngle[i] ,  azimuthAngle = -256)
    lat_ipp_4.append(lat_ipp_i)
    lon_ipp_4.append(lon_ipp_i)
    

inc_angle = 42
inc_angle_iono = iono.incidence_angle_ground2iono(inc_angle)

r_iono_1 = [[] for _ in range(len(tec_maps))]
for k in trange(len(tec_maps), desc='Outer Processing'):
    r_iono = []
    for i in trange(len(ascending_frame_time_utc_sec), desc='Inner Processing'):
        try:
            from datetime import datetime
            result = ionex.interp_3d_maps(tec_maps_array[k],mins ,lats, lons , ascending_frame_time_utc_sec[i] , lat_ipp_1[i], lon_ipp_1[i] , interp_method='linear3d', rotate_tec_map=True, print_msg=True)
            result = iono.vtec2range_delay(result, inc_angle=inc_angle_iono, freq=iono.SAR_BAND['L'])
        except Exception as e:
            print(f"Error occurred: {e}")
            result = np.nan
        r_iono.append(result)
    r_iono_1[k] = r_iono

r_iono_2 = [[] for _ in range(len(tec_maps))]
for k in trange(len(tec_maps), desc='Outer Processing'):
    r_iono = []
    for i in trange(len(ascending_frame_time_utc_sec), desc='Inner Processing'):
        try:
            from datetime import datetime
            result = ionex.interp_3d_maps(tec_maps_array[k],mins ,lats, lons , ascending_frame_time_utc_sec[i] , lat_ipp_2[i], lon_ipp_2[i] , interp_method='linear3d', rotate_tec_map=True, print_msg=True)
            result = iono.vtec2range_delay(result, inc_angle=inc_angle_iono, freq=iono.SAR_BAND['L'])
        except Exception as e:
            print(f"Error occurred: {e}")
            result = np.nan
        r_iono.append(result)
    r_iono_2[k] = r_iono

r_iono_3 = [[] for _ in range(len(tec_maps))]
for k in trange(len(tec_maps), desc='Outer Processing'):
    r_iono = []
    for i in trange(len(ascending_frame_time_utc_sec), desc='Inner Processing'):
        try:
            from datetime import datetime
            result = ionex.interp_3d_maps(tec_maps_array[k],mins ,lats, lons , ascending_frame_time_utc_sec[i] , lat_ipp_3[i], lon_ipp_3[i] , interp_method='linear3d', rotate_tec_map=True, print_msg=True)
            result = iono.vtec2range_delay(result, inc_angle=inc_angle_iono, freq=iono.SAR_BAND['L'])
        except Exception as e:
            print(f"Error occurred: {e}")
            result = np.nan
        r_iono.append(result)
    r_iono_3[k] = r_iono

r_iono_4 = [[] for _ in range(len(tec_maps))]
for k in trange(len(tec_maps), desc='Outer Processing'):
    r_iono = []
    for i in trange(len(ascending_frame_time_utc_sec), desc='Inner Processing'):
        try:
            from datetime import datetime
            result = ionex.interp_3d_maps(tec_maps_array[k],mins ,lats, lons , ascending_frame_time_utc_sec[i] , lat_ipp_4[i], lon_ipp_4[i] , interp_method='linear3d', rotate_tec_map=True, print_msg=True)
            result = iono.vtec2range_delay(result, inc_angle=inc_angle_iono, freq=iono.SAR_BAND['L'])
        except Exception as e:
            print(f"Error occurred: {e}")
            result = np.nan
        r_iono.append(result)
    r_iono_4[k] = r_iono


with h5py.File(f'data/global_a/asc_iono_{year}.h5', 'w') as h5f:
        # 将数据写入.h5文件
    h5f.create_dataset('r_iono_1', data=r_iono_1)
    h5f.create_dataset('r_iono_2', data=r_iono_2)
    h5f.create_dataset('r_iono_3', data=r_iono_3)
    h5f.create_dataset('r_iono_4', data=r_iono_4)
































