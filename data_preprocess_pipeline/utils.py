from datetime import datetime
import pandas as pd
from libcomcat.search import search
import os
import requests
import numpy as np
import pickle
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from dtaidistance import dtw
from scipy.signal import savgol_filter
from joblib import Parallel, delayed


def download_earthquake_data(start_year, end_year, minlatitude, maxlatitude, minlongitude, maxlongitude,save_path,minmagnitude=2.5):
    os.makedirs(save_path, exist_ok=True)
    for year in range(start_year, end_year + 1):
        box_events = search(starttime=datetime(year, 1, 1, 00, 00), endtime=datetime(year+1, 1, 1, 00, 00),
                    minlatitude=minlatitude, maxlatitude=maxlatitude, minlongitude=minlongitude, maxlongitude=maxlongitude,
                    minmagnitude=minmagnitude, maxmagnitude=10)
        events_data = [{
        'ID': event.id,
        'Time': event.time,
        'Magnitude': event.magnitude,
        'Latitude': event.latitude,
        'Longitude': event.longitude,
        'Depth': event.depth
        } for event in box_events]
        df = pd.DataFrame(events_data)
        df.to_csv(save_path+'/earthquakes_{}.csv'.format(year), index=False)
        print('Year {} has {} events'.format(year, len(events_data)))


def download_GNSS_data(station_names, save_path):
    # 定义文件下载函数
    def download_file(file_url, download_path):
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"下载成功: {file_url}")
        else:
            print(f"下载失败: {file_url}, 状态码: {response.status_code}")
    os.makedirs(save_path, exist_ok=True)
    # 循环下载每个站点的 .tenv3 文件
    for station_name in station_names:
        url = "http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/" + station_name + ".tenv3"

        download_path = os.path.join(save_path, station_name)
        if os.path.exists(download_path):
            print(f"已经下载过: {station_name}")
            continue
        
        download_file(url, download_path)



def get_use_station_dict(station_dict_all,maxlatitude,minlatitude,minlongitude,maxlongitude):
    station_dict_use = {}
    for station_name, station_dict in station_dict_all.items():
        if station_dict['latitude'] >= minlatitude and station_dict['latitude'] <= maxlatitude and station_dict['longitude'] >= minlongitude and station_dict['longitude'] <= maxlongitude:
            station_dict_use[station_name] = station_dict
    return station_dict_use



def process_usgs_data(input_dir, output_dir, lat_min=32, lat_max=36, lon_min=-120, lon_max=-114, lat_step=0.1, lon_step=0.1, topk=None):
    """
    处理USGS地震数据，按经纬度将数据网格化，并仅保存地震发生次数最多的前 topk 个网格的数据，且为这些网格分别设置编号为 0 到 topk-1。

    参数：
    - input_dir: 包含原始数据的目录路径。
    - output_dir: 保存网格化数据的目录路径。
    - lat_min: 纬度最小值。
    - lat_max: 纬度最大值。
    - lon_min: 经度最小值。
    - lon_max: 经度最大值。
    - lat_step: 纬度步长。
    - lon_step: 经度步长。
    - topk: 保留地震次数最多的前 topk 个网格。如果为 None，则保留所有网格。
    """
    # 列出目录下的文件
    df_list = sorted(os.listdir(input_dir))
    df_list = [x for x in df_list if x.endswith('.csv')]

    # 生成纬度和经度的分箱
    lat_bins = np.around(np.linspace(lat_min, lat_max, int((lat_max - lat_min) / lat_step) + 1), 2)
    lon_bins = np.around(np.linspace(lon_min, lon_max, int((lon_max - lon_min) / lon_step) + 1), 2)

    # 初始化网格
    grid = {}
    for lat in lat_bins[:-1]:
        for lon in lon_bins[:-1]:
            grid[(lat, lon)] = []

    # 遍历每个文件并处理数据
    for df_dir in df_list:
        df = pd.read_csv(os.path.join(input_dir, df_dir))
        df['LatBin'] = pd.cut(df['Latitude'], bins=lat_bins, labels=lat_bins[:-1])
        df['LonBin'] = pd.cut(df['Longitude'], bins=lon_bins, labels=lon_bins[:-1])
        # 处理每一行数据
        for index, row in df.iterrows():
            lat_bin = row['LatBin']
            lon_bin = row['LonBin']
            if pd.notna(lat_bin) and pd.notna(lon_bin):
                grid[(lat_bin, lon_bin)].append({
                    'ID': row['ID'],
                    'Time': row['Time'],
                    'Magnitude': row['Magnitude'],
                    'Latitude': row['Latitude'],
                    'Longitude': row['Longitude'],
                    'Depth': row['Depth']
                })
        # 不再保存标注后的数据到 input_dir

    # 根据地震次数对网格进行排序
    grid_counts = {key: len(events) for key, events in grid.items()}
    sorted_grid = sorted(grid_counts.items(), key=lambda item: item[1], reverse=True)

    # 如果设置了 topk，则只保留前 topk 个网格
    if topk is not None:
        sorted_grid = sorted_grid[:topk]

    # 重新为前 topk 个网格设置编号为 0 到 topk-1
    grid_id_map = {sorted_grid[i][0]: i for i in range(len(sorted_grid))}
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    grid_dict = {}
    # 保存每个网格的数据并标注新的编号
    for key, _ in sorted_grid:
        events = grid[key]
        # 为每个事件添加 'Location_id'
        for event in events:
            event['Location_id'] = grid_id_map[key]
        df_grid = pd.DataFrame(events)
        file_path = os.path.join(output_dir, f'grid_{np.around(key[0], 2)}_{np.around(key[1], 2)}.csv')
        df_grid.to_csv(file_path, index=False)
        grid_dict[grid_id_map[key]] = key
    # 保存网格编号映射
    with open(os.path.join(output_dir, 'grid_id_map.pkl'), 'wb') as f:
        pickle.dump(grid_dict, f)

    print(f"所有前 {topk} 个网格数据已保存，并为这些网格设置了 0 到 {topk-1} 的编号。")
    return grid_dict


def process_energy_data(input_dir, output_dir, start_date='1986-01-01', end_date='2024-08-31', freq='2W'):
    """
    处理地震网格数据，计算每个时间段的对数能量并保存到指定目录。

    参数：
    - input_dir: 包含网格数据的输入目录路径。
    - output_dir: 计算对数能量数据的保存目录路径。
    - start_date: 起始时间，默认为'1986-01-01'。
    - end_date: 结束时间，默认为'2024-08-31'。
    - freq: 时间间隔的频率，默认为'2W'（每两周）。
    """
    # 列出目录下的文件
    data_list = os.listdir(input_dir)
    data_list = [x for x in data_list if x.endswith('.csv')]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历每个文件并处理数据
    count = 0
    for data_dir in data_list:
        file_path = os.path.join(input_dir, data_dir)

        
        # 检查文件是否存在且非空
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                data = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                continue
        
            # 处理时间数据
            data['Time'] = pd.to_datetime(data['Time'], format='ISO8601', errors='coerce')
            data.set_index('Time', inplace=True)
            data = data.dropna(subset=['Magnitude'])  # 移除无效的震级数据

            # 生成时间区间
            time_bins = pd.date_range(start=pd.Timestamp(start_date, tz='UTC'), 
                                      end=pd.Timestamp(end_date, tz='UTC'), 
                                      freq=freq)
            
            # 将数据按时间区间分箱
            data['Time_bin'] = pd.cut(data.index, bins=time_bins, right=True)
            
            # 计算每个时间段的能量
            def calculate_energy(group):
                return np.sum(10 ** (1.5 * group['Magnitude']))
            
            grouped_energy = data.groupby('Time_bin', observed=False).apply(lambda x: calculate_energy(x))

            # 计算对数能量并填充空值
            log_energy = ((1 / 1.5) * np.log10(grouped_energy.replace(0, np.nan))).replace(np.nan, 0)
            log_energy_filled = pd.DataFrame()
            log_energy_filled["Energy"] = log_energy.reindex(time_bins[:-1], fill_value=0)
            log_energy_filled["Location_id"] = data['Location_id'].iloc[0]
            log_energy_filled.index = log_energy_filled.index.strftime('%Y-%m-%d')
            
            # 保存处理后的数据
            log_energy_filled.to_csv(os.path.join(output_dir, data_dir + ".csv"), index=True)
            count += 1

    print(f"所有数据已保存, 共有 {count} 个网格数据被处理。")


def construct_earthquake_csv(directory, start_date, end_date, output_file):
    """
    构建包含所有地震站点的最大震级数据的CSV文件。

    参数：
    - directory: 包含地震数据的目录路径。
    - start_date: 数据的起始日期。
    - end_date: 数据的结束日期。
    - output_file: 输出的CSV文件路径。
    """
    data_frames = []
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    file_dir = os.listdir(directory)
    file_dir = [x for x in file_dir if x.endswith('.csv')]
    for station_file in file_dir:
        df_temp = pd.DataFrame(index=date_range)
        file_path = os.path.join(directory, station_file)
        
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                df = pd.read_csv(file_path)
                if "Location_id" in df.columns and "Magnitude" in df.columns:
                    Location_id = df["Location_id"].iloc[0]
                    # 处理时间并聚合震级数据
                    df['Time'] = pd.to_datetime(df['Time'], format='ISO8601').dt.date
                    df = df.groupby('Time').agg({'Magnitude': 'max'}).reset_index()
                    df.set_index('Time', inplace=True)
                    magnitude_series = df['Magnitude'].reindex(date_range).fillna(0)
                    df_temp[Location_id] = magnitude_series
                    data_frames.append(df_temp)
            except pd.errors.EmptyDataError:
                continue

    # 合并所有站点数据，按列排序
    df = pd.concat(data_frames, axis=1)
    df = df.sort_index(axis=1)
    df.to_csv(output_file)
    return df


def construct_energy_csv(directory, output_file):
    """
    构建包含所有地震站点的能量数据的CSV文件。

    参数：
    - directory: 包含能量数据的目录路径。
    - output_file: 输出的CSV文件路径。
    """
    data_frames = []

    for station_file in os.listdir(directory):
        file_path = os.path.join(directory, station_file)
        df_temp = pd.DataFrame()
        
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                df = pd.read_csv(file_path, index_col=0)
                if "Location_id" in df.columns and "Energy" in df.columns:
                    Location_id = df["Location_id"].iloc[0]
                    df_temp[Location_id] = df['Energy']
                    df_temp.index = df.index
                    data_frames.append(df_temp)
            except pd.errors.EmptyDataError:
                continue
    
    # 合并所有站点数据，按列排序
    print("Length of energy_frames: ", len(data_frames))
    df = pd.concat(data_frames, axis=1)
    df = df.sort_index(axis=1)
    df.to_csv(output_file)
    return df


def has_consecutive_nans(series, window=30):
    """
    检查给定序列中是否存在连续的缺失值。

    参数:
    series (pd.Series): 要检查的序列。
    window (int): 连续缺失值的窗口大小。

    返回:
    bool: 如果存在连续的缺失值，则返回 True，否则返回 False。
    """
    return series.isna().rolling(window=window).apply(lambda x: x.sum() >= window).any()

def construct_gnss_csv(directory, start_date, end_date, output_file, station_dict):
    """
    构建包含所有GNSS站点对齐后的CSV文件，并处理缺失数据。

    参数:
    - directory: 包含GNSS日数据的输入目录路径。
    - start_date: 起始日期。
    - end_date: 结束日期。
    - output_file: 输出的CSV文件路径。
    """
    # 初始化时间范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data_frames = []
    station_dict_fliter = {}
    for station_file in os.listdir(directory):
        file_path = os.path.join(directory, station_file)
        
        # 检查文件是否存在且非空
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                # 读取文件，假定数据以空格分隔
                data_day = pd.read_csv(file_path, sep='\s+')
                data_day['Date'] = pd.to_datetime(data_day['YYMMMDD'], format='%y%b%d')
                
                # 只保留start_date之后的数据
                data_day = data_day[data_day['Date'] >= start_date]

                # 合并北向、东向和高程方向上的位移，并计算总位移
                data_day['combined'] = data_day.apply(lambda row: [row['_north(m)'], row['__east(m)'], row['____up(m)'], np.sqrt(row['__east(m)']**2 + row['_north(m)']**2 + row['____up(m)']**2)], axis=1)
                station_name = os.path.splitext(station_file)[0]
                temp_df = pd.DataFrame(data_day['combined'].values, columns=[station_name], index=data_day['Date'])
                # 将数据对齐到统一的日期范围
                temp_df = temp_df.reindex(date_range)

                data_frames.append(temp_df)
                print(f"监测站 {station_name} 的数据已添加。")
                station_dict_fliter[station_name] = station_dict[station_name]
                
            except pd.errors.EmptyDataError:
                print(f"文件 {station_file} 为空，已被忽略。")
                continue

    # 合并所有监测站的数据，按时间对齐
    if data_frames:
        df = pd.concat(data_frames, axis=1)
        df.to_csv(output_file)
        print(f"CSV文件已创建：{output_file}")
    else:
        print("没有有效的监测站数据。")
    return station_dict_fliter

def generate_time_bins(start_date, end_date, time_resolution=14):
    """
    高效批量生成以time_resolution天为间隔的时间窗口。

    参数:
        start_date (str or pd.Timestamp): 开始日期。
        end_date (str or pd.Timestamp): 结束日期。
        time_resolution (int): 间隔的天数。

    返回:
        pd.DatetimeIndex: 以time_resolution天为间隔的时间窗口序列。
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    time_bins = pd.date_range(start=start_date, end=end_date + pd.Timedelta(days=time_resolution), freq=f'{time_resolution}D')
    return time_bins

def no_lag_filter(data, window_length=9, polyorder=3):
    if window_length % 2 == 0:
        raise ValueError("窗口长度必须是奇整数。")
    smoothed_data = savgol_filter(data, window_length=window_length, polyorder=polyorder)
    return smoothed_data

def generate_sem_aij(data, output_file, window_length=9, polyorder=3):
    df = calculate_energy_in_time_window(data, time_resolution = 14)
    n_columns = df.shape[1]
    similarity_matrix = np.zeros((n_columns, n_columns))

    # 预处理每列数据，使用无滞后滤波器
    filtered_data = df.apply(lambda col: no_lag_filter(col.values, window_length, polyorder), axis=0)

    def compute_distance(i, j):
        distance = dtw.distance(filtered_data.iloc[:, i].values, filtered_data.iloc[:, j].values)
        print(f"Computing distance between columns {i} and {j}: {distance}")
        return i, j, distance

    # 并行计算距离，并收集结果
    results = Parallel(n_jobs=-1)(
        delayed(compute_distance)(i, j) 
        for i in range(n_columns) 
        for j in range(i, n_columns)
    )

    for i, j, distance in results:
        similarity_matrix[i, j] = distance
        similarity_matrix[j, i] = distance  # 对称矩阵
    Aij_df = pd.DataFrame(similarity_matrix, index=df.columns, columns=df.columns)
    Aij_df.to_csv(output_file)
    return Aij_df


from math import radians, sin, cos, sqrt, atan2

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度点之间的地球表面距离，单位为公里。
    
    参数:
    - lat1, lon1: 第一个点的纬度和经度。
    - lat2, lon2: 第二个点的纬度和经度。
    
    返回:
    - 距离，单位为公里。
    """
    # 将经纬度从度转换为弧度
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine公式计算距离
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371  # 地球半径，单位为公里
    return r * c

def generate_geo_aij(station_dict,save_path):
    """
    基于站点之间的距离构建邻接矩阵。距离超过阈值的站点将不连接。
    
    参数:
    - station_dict: 包含站点名和经纬度的字典，格式为 {站点名: (纬度, 经度)}。
    - distance_threshold: 站点之间距离的阈值，超过该距离的站点不连接。
    
    返回:
    - 邻接矩阵Aij (pd.DataFrame)。
    """
    station_names = list(station_dict.keys())
    num_stations = len(station_names)
    
    # 初始化邻接矩阵
    Aij = np.zeros((num_stations, num_stations))

    # 计算所有站点之间的距离
    for i, station1 in enumerate(station_names):
        for j, station2 in enumerate(station_names):
            if i != j:
                lat1, lon1 = station_dict[station1]
                lat2, lon2 = station_dict[station2]
                lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                
                Aij[i, j] = distance
                Aij[j, i] = distance

    Aij_df = pd.DataFrame(Aij, index=station_names, columns=station_names)
    Aij_df.to_csv(save_path)
    return Aij_df


def calculate_energy_in_time_window(data, time_resolution=14):
    """
    计算在指定时间窗口内的能量。

    参数:
        data (pd.DataFrame): 包含站点数据的DataFrame，行名为日期，列名为站点名。
        time_resolution (int): 时间窗口的间隔天数。

    返回:
        pd.DataFrame: 每个站点在每个时间窗口内的对数能量结果。
    """
    # 自动获取开始和结束日期
    start_date = data.index.min()
    end_date = data.index.max()

    time_bins = generate_time_bins(start_date, end_date, time_resolution=time_resolution)

    data = data.copy()
    # 将日期分配到时间窗口
    data['Time_bin'] = pd.cut(data.index, bins=time_bins, right=False)

    numeric_cols = data.select_dtypes(include=[np.number]).columns

    data[numeric_cols] = data[numeric_cols].where(data[numeric_cols] > 0)

    # 先对每个震级计算能量
    data[numeric_cols] = 10 ** (1.5 * data[numeric_cols])

    # 按时间窗口分组，并对能量求和
    grouped_energy = data.groupby('Time_bin', observed=True)[numeric_cols].sum()

    # 计算对数能量
    log_energy = (1 / 1.5) * np.log10(grouped_energy.replace(0, np.nan))

    # 填充NaN为0
    log_energy_filled = log_energy.fillna(0)

    # 更新索引为时间窗口的开始日期
    log_energy_filled.index = log_energy_filled.index.map(lambda x: x.left.strftime('%Y-%m-%d'))

    return log_energy_filled