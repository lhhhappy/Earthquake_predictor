import pandas as pd
import pickle 
import os
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import ast
import torch
from torch.utils.data import Dataset
from scipy.linalg import eigh
from torch.nn import functional as F
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import copy

def normalize(data, method='min-max'):
    """
    Normalize the input data independently for each N and C using vectorized operations.
    
    Parameters:
        data (np.ndarray): Input data of shape (T, N, C).
        method (str): Normalization method. Options are 'z-score', 'min-max', 'max'.
        
    Returns:
        np.ndarray: Normalized data with the same shape as the input.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a numpy array.")
    if data.ndim != 3:
        raise ValueError("Input data must have 3 dimensions: (T, N, C).")
    
    if method == 'z-score':
        # Compute mean and std along the T dimension, keep dimensions for broadcasting
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        normalized_data = (data - mean) / std

    elif method == 'min-max':
        # Compute min and max along the T dimension
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        normalized_data = (data - min_val) / range_val

    elif method == 'max':
        # Compute max along the T dimension
        max_val = np.max(data, axis=0, keepdims=True)
        # Avoid division by zero
        max_val = np.where(max_val == 0, 1, max_val)
        normalized_data = data / max_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized_data
def dataframe_to_array(df):
    df_filled = df.apply(lambda col: col.map(lambda x: x if isinstance(x, list) and len(x) == 4 else [np.nan] * 4))
    array_3d = np.array(df_filled.values.tolist()).reshape(df.shape[0], df.shape[1], 4)
    return array_3d

def Trend_filter(data, window_length=9, polyorder=3, axis=0):
    """
    应用Savitzky-Golay滤波器在指定轴上平滑数据而不引入滞后。

    :param data: 数值数据点的NumPy数组。
    :param window_length: 滤波窗口的长度（必须是奇整数）。
    :param polyorder: 用于拟合样本的多项式的阶数。
    :param axis: 要应用滤波器的轴。
    :return: 平滑后的NumPy数组。
    """
    if window_length % 2 == 0:
        raise ValueError("窗口长度必须是奇整数。")

    # 获取指定轴上的大小
    size_along_axis = data.shape[axis]
    # 如有必要，调整窗口长度
    if size_along_axis < window_length:
        window_length = size_along_axis if size_along_axis % 2 != 0 else size_along_axis - 1
        
        if window_length < polyorder + 2:
            raise ValueError("数据点不足以应用滤波器。")

    # 在指定轴上应用Savitzky-Golay滤波器
    smoothed_data = savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=axis)
    return smoothed_data

def apply_filter_to_all_nodes(data_array, window_length=5, polyorder=3):
    """
    对数据数组中的所有节点和通道的T维度应用Trend_filter。

    :param data_array: 形状为(N, T, C)的NumPy数组。
    :param window_length: 滤波窗口的长度（必须是奇整数）。
    :param polyorder: 用于拟合样本的多项式的阶数。
    :return: 具有与data_array相同形状的NumPy数组，包含平滑后的数据。
    """
    T, N, C = data_array.shape

    # 如果T小于窗口长度，则调整窗口长度
    if T < window_length:
        adjusted_window_length = T if T % 2 != 0 else T - 1
        if adjusted_window_length < polyorder + 2:
            window_length = adjusted_window_length

    # 在T轴（axis=0）上应用Trend_filter
    filtered_data = Trend_filter(data_array, window_length=window_length, polyorder=polyorder, axis=0)

    return filtered_data

def parse_str_list(cell):
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return cell
    
def get_gnss_i(series, index):
    #[n, e, u, vectors]
    return series.apply(lambda x: x[index] if isinstance(x, list) and len(x) > index else None)


class EarthquakeGNSSDataset(Dataset):
    def __init__(self, 
                 area,
                 earthquake_data, gnss_data,
                 es_geo_matrix, es_sem_matrix, gnss_geo_matrix,
                 geo_percentage, sem_percentage, lape_dim, earthquake_dict_use, station_dict_use,
                 window_size=14, forecast_horizon=14, time_resolution=14, earthquake_catalog_window=1400,
                 earthquake_threshold=4.0, missing_threshold=5):
        """
        地震-GNSS数据集的自定义Dataset类。

        参数：
        - area: 区域名称。
        - earthquake_data: 包含地震数据的DataFrame。
        - gnss_data: 包含GNSS数据的DataFrame。
        - es_geo_matrix: 地震站点的地理邻接矩阵。
        - gnss_geo_matrix: GNSS站点的地理邻接矩阵。
        - lape_dim: 图拉普拉斯嵌入的维度数。
        - window_size: 历史窗口大小，默认14。
        - forecast_horizon: 预测窗口大小，默认14。
        - earthquake_catalog_window: 地震历史窗口大小，默认1400。
        - earthquake_threshold: 地震震级阈值，默认4.0。
        - missing_threshold: 允许的最大连续缺失值数量，默认5。
        """
        self.area = area
        self.earthquake_data = earthquake_data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.earthquake_threshold = earthquake_threshold
        self.earthquake_window = earthquake_catalog_window
        self.missing_threshold = missing_threshold
        self.lape_dim = lape_dim
        self.time_resolution = time_resolution
        self.earthquake_location = torch.tensor([earthquake_dict_use[i] for i in sorted(earthquake_dict_use.keys())], dtype=torch.float32)

        self.station_location = np.array([station_dict_use[i] for i in sorted(station_dict_use.keys())], dtype=np.float32)

        self.earthquake_data_date = earthquake_data.index
        self.gnss_data_date = gnss_data.index

        self.gnss_data = dataframe_to_array(gnss_data)

        # 生成掩码
        self.es_geo_mask, self.es_sem_mask, self.gnss_geo_mask = generate_masks(
            es_geo_matrix, es_sem_matrix, gnss_geo_matrix, geo_percentage, sem_percentage
        )
        self.start_index = self.find_first_valid_index(self.gnss_data)

        # 设置起始时间索引
        start_day = pd.Timestamp(gnss_data.index[self.start_index].date())
        self.earthquake_start_index = earthquake_data.index.get_loc(start_day)

        # 确保在地震数据和 GNSS 数据长度的基础上计算数据集长度
        self.length = min(
            len(gnss_data) - self.start_index - self.window_size - self.forecast_horizon + 1,
            len(earthquake_data) - self.earthquake_start_index - self.window_size - self.forecast_horizon
        )
        

        self.lap_ex = graph_laplacian_embedding(torch.tensor(es_geo_matrix.values), lape_dim)

    def __len__(self):
        return self.length

    def __name__(self):
        return self.area

    def __getitem__(self, idx):
        # 获取历史和未来的地震数据
        earthquake_data_history = self.earthquake_data.iloc[
            max(0, self.earthquake_start_index + self.window_size + idx - self.earthquake_window):
            self.earthquake_start_index + self.window_size + idx
        ]
        earthquake_data_future = self.earthquake_data.iloc[
            self.earthquake_start_index + self.window_size + idx:
            self.earthquake_start_index + self.window_size + idx + self.forecast_horizon
        ]

        # 获取GNSS历史数据并转置以匹配期望的维度
        gnss_data_history = self.gnss_data[
            idx + self.start_index:idx + self.window_size + self.start_index
        ].transpose(1, 0, 2)

        
        
        # earthquake_data_history_date = self.earthquake_data_date[
        #     max(0, self.earthquake_start_index + self.window_size + idx - self.earthquake_window):
        #     self.earthquake_start_index + self.window_size + idx
        # ]

        # earthquake_data_future_date = self.earthquake_data_date[
        #     self.earthquake_start_index + self.window_size + idx:
        #     self.earthquake_start_index + self.window_size + idx + self.forecast_horizon
        # ]

        # gnss_data_history_date = self.gnss_data_date[
        #     idx + self.start_index:idx + self.window_size + self.start_index
        # ]
        # print("earthquake_start",earthquake_data_history_date[0])
        # print("earthquake_end",earthquake_data_history_date[-1])
        # print("gnss_start",gnss_data_history_date[0])
        # print("gnss_end",gnss_data_history_date[-1])
        # print("earthquake_future_start",earthquake_data_future_date[0])
        # print("earthquake_future_end",earthquake_data_future_date[-1]) 

        
        earthquake_happen = torch.tensor((earthquake_data_future >= self.earthquake_threshold).any(axis=0).to_numpy(), dtype=torch.bool)

        # 计算历史和未来的对数能量
        log_energy_history = calculate_energy_in_time_window(earthquake_data_history,self.time_resolution).values.T
        log_energy_future = calculate_energy_in_time_window(earthquake_data_future,self.time_resolution).values.T

        # 获取未来地震事件发生的天数
        earthquake_data_future_day = find_first_earthquake(
            earthquake_data_future, self.earthquake_threshold
        )
        
        # 检测并移除具有长连续缺失数据的站点
        missing_mask = np.isnan(gnss_data_history).all(axis=2)
        max_missing_lengths = self.max_consecutive_trues(missing_mask)
        stations_to_keep = max_missing_lengths <= self.missing_threshold

        # 更新gnss_data_history和gnss_geo_mask
        gnss_data_history = gnss_data_history[stations_to_keep]
        station_location_use = self.station_location[stations_to_keep]

        sample_gnss_geo_mask = self.gnss_geo_mask[np.ix_(stations_to_keep, stations_to_keep)]

        # 生成GNSS数据的图拉普拉斯嵌入
        sample_lap_gnss = graph_laplacian_embedding(sample_gnss_geo_mask.float(), self.lape_dim)

        # 填充缺失值
        gnss_data_history = fill_nan_with_interpolation(gnss_data_history).transpose(1, 0, 2)

        gnss_data_history = normalize(gnss_data_history, method='min-max')
        
        gnss_trend_filtered = apply_filter_to_all_nodes(gnss_data_history, window_length = 21 , polyorder=3)
        gnss_seasonal_component = gnss_data_history - gnss_trend_filtered
        
        gnss_data_history = np.concatenate([gnss_trend_filtered, gnss_seasonal_component], axis=2)
        # 将数据转换为张量
        log_energy_history = torch.tensor(
            log_energy_history, dtype=torch.float32
        ).permute(1, 0).unsqueeze(-1)
        log_energy_future = torch.tensor(log_energy_future, dtype=torch.float32)
        gnss_data_history = torch.tensor(gnss_data_history, dtype=torch.float32)
        earthquake_data_future_day = torch.tensor(
            earthquake_data_future_day, dtype=torch.float32
        ).squeeze(-1)
        station_location_use = torch.tensor(station_location_use, dtype=torch.float32)

        return {
            'log_energy_history': log_energy_history,
            'gnss_data_history': gnss_data_history,
            'log_energy_future': log_energy_future,
            'earthquake_data_future_day': earthquake_data_future_day,
            'es_geo_mask': self.es_geo_mask,
            'es_sem_mask': self.es_sem_mask,
            'gnss_geo_mask': sample_gnss_geo_mask,
            'lap_ex': self.lap_ex,
            'lap_gnss': sample_lap_gnss,
            "earthquake_happen":earthquake_happen,
            "earthquake_location":self.earthquake_location,
            "station_location":station_location_use
        }
    def find_first_valid_index(self, gnss_data):
        """
        查找 GNSS 数据中第一个包含至少两个非 NaN 数据的时间点索引。
        """
        for idx, data in enumerate(gnss_data):
            # 统计非 NaN 元素数量
            if np.sum(~np.isnan(data)) >= 5:
                return idx
        return 
    
    @staticmethod
    def max_consecutive_trues(arr):
        """
        高效计算每行最大连续True的数量。

        参数：
        - arr: 布尔型numpy数组，形状为(num_stations, window_size)

        返回：
        - 每个站点的最大连续True计数，形状为(num_stations,)
        """
        # 将布尔值转换为整数
        arr_int = arr.astype(int)
        # 在每行的开头和结尾添加零
        padded = np.pad(arr_int, ((0, 0), (1, 1)), mode='constant', constant_values=0)
        # 计算差分
        diff = np.diff(padded, axis=1)
        # 找到连续True的开始和结束索引
        run_starts = np.where(diff == 1)
        run_ends = np.where(diff == -1)
        # 计算连续True的长度
        run_lengths = run_ends[1] - run_starts[1]
        # 初始化最大长度数组
        max_lengths = np.zeros(arr.shape[0], dtype=int)
        # 使用np.maximum.at更新每个站点的最大连续True长度
        np.maximum.at(max_lengths, run_starts[0], run_lengths)
        return max_lengths

# 支持函数
def has_consecutive_nans(series, window=30):
    """
    检查序列中是否存在连续缺失值。

    参数:
    - series (pd.Series): 要检查的序列。
    - window (int): 连续缺失值的窗口大小。

    返回:
    - bool: 是否存在连续的缺失值。
    """
    return series.isna().rolling(window=window).sum().max() >= window

def convert_to_fixed_length_array(data, length=4):
    """
    将数据转换为固定长度的数组。

    参数：
    - data: 输入数据，类型为数组或列表，长度为window_size。
    - length: 目标长度。

    返回：
    - 形状为(window_size, length)的numpy数组。
    """
    # data是长度为window_size的数组，包含每个时间步的数据
    fixed_length_array = []
    for x in data:
        if isinstance(x, (list, np.ndarray)):
            # 截取或填充x以达到固定长度
            x = x[:length]
            if len(x) < length:
                x = list(x) + [np.nan] * (length - len(x))
        else:
            # 如果x不是列表或数组，用NaN填充
            x = [np.nan] * length
        fixed_length_array.append(x)
    return np.array(fixed_length_array)  # 形状：(window_size, length)

def fill_nan_with_interpolation(data):

    """
    使用插值填充数据中的NaN值。

    参数：
    - data: 形状为(num_stations, window_size, num_features)的numpy数组

    返回：
    - 填充NaN后的数据
    """

    num_stations, window_size, num_features = data.shape
    # 将数据展平成二维数组，形状为(num_stations * num_features, window_size)
    data_reshaped = data.transpose(0, 2, 1).reshape(num_stations * num_features, window_size)
    # 创建缺失值掩码
    nans = np.isnan(data_reshaped)
    # 对于每一行（对应一个特征的时间序列），进行插值
    for i in range(data_reshaped.shape[0]):
        if not nans[i].all():
            data_reshaped[i][nans[i]] = np.interp(
                np.flatnonzero(nans[i]), np.flatnonzero(~nans[i]), data_reshaped[i][~nans[i]]
            )
        else:
            # 如果整行都是NaN，用零替换
            data_reshaped[i] = np.zeros(window_size)
    # 恢复到原始形状，并转置回 (num_stations, window_size, num_features)
    data_filled = data_reshaped.reshape(num_stations, num_features, window_size).transpose(0, 2, 1)
    
    return data_filled

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

def find_first_earthquake(earthquake_catalog, threshold):
    """
    返回每个站点的第一个大于阈值的地震发生的天数（行数）。

    参数:
    - earthquake_catalog (pd.DataFrame): 地震目录，每个值为地震震级，行名为日期。
    - threshold (float): 震级��值。

    返回:
    - result_vector (np.ndarray): (num_stations, 1)的向量，表示第一个超过阈值的地震发生的行数。
    """
    earthquake_data = earthquake_catalog.to_numpy()
    # 创建布尔矩阵，标记超过阈值的事件
    above_threshold = earthquake_data > threshold
    # 初始化结果向量
    result_vector = np.zeros((earthquake_data.shape[1], 1))
    # 找到第一个超过阈值的索引
    first_event_indices = np.argmax(above_threshold, axis=0)
    # 检查是否有任何事件
    has_event = above_threshold.any(axis=0)
    # 更新结果向量
    result_vector[has_event, 0] = first_event_indices[has_event] + 1

    return result_vector

def graph_laplacian_embedding(adj_matrix, k):
    """
    计算图的拉普拉斯嵌入。

    参数:
    - adj_matrix (torch.Tensor): 图的邻接矩阵。
    - k (int): 选择的特征向量数量。

    返回:
    - torch.Tensor: 形状为(N, k)的嵌入矩阵。
    """
    adj_matrix = adj_matrix.float()
    N = adj_matrix.shape[0]
    
    # 特殊情况：如果邻接矩阵是1x1，直接返回一个值为0的1x1矩阵
    if N == 1:
        return torch.zeros((1, 1), device=adj_matrix.device)

    k = min(k, N - 1)

    # 计算度矩阵
    degrees = torch.sum(adj_matrix, dim=1)
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees + 1e-8))
    
    # 计算归一化的拉普拉斯矩阵
    laplacian = torch.eye(N, device=adj_matrix.device) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    
    # 特征分解
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    
    # 选择k个最小的非平凡特征向量
    selected_eigenvectors = eigenvectors[:, 1:k+1]

    return selected_eigenvectors

def normalize_locations(quake_location, station_location):
    """
    Normalize latitude and longitude for quake and station locations to range [-1, 1].

    Args:
    quake_location (torch.Tensor): Tensor of shape (num, 2) for earthquake locations (latitude, longitude).
    station_location (torch.Tensor): Tensor of shape (num, 2) for station locations (latitude, longitude).

    Returns:
    torch.Tensor, torch.Tensor: Normalized quake and station locations.
    """
    # Clone the input tensors to avoid modifying the original ones
    normalized_quake_location = quake_location.clone()
    normalized_station_location = station_location.clone()

    # Normalize quake locations
    normalized_quake_location[:, 0] = 2 * (normalized_quake_location[:, 0] + 90) / 180 - 1  # Latitude normalization
    normalized_quake_location[:, 1] = 2 * (normalized_quake_location[:, 1] + 180) / 360 - 1  # Longitude normalization

    # Normalize station locations
    normalized_station_location[:, 0] = 2 * (normalized_station_location[:, 0] + 90) / 180 - 1  # Latitude normalization
    normalized_station_location[:, 1] = 2 * (normalized_station_location[:, 1] + 180) / 360 - 1  # Longitude normalization

    return normalized_quake_location, normalized_station_location

def generate_masks(es_geo_matrix, es_sem_matrix, gnss_geo_matrix, geo_percentage=0.3, sem_percentage=0.3):
    """
    生成地理和语义掩码，使用百分比来定义掩码阈值。

    参数:
    - es_geo_matrix: 地震-站点的地理邻接矩阵。
    - es_sem_matrix: 地震-站点的语义邻接矩阵。
    - gnss_geo_matrix: GNSS邻接矩阵。
    - geo_percentage: 需要掩蔽的地理距离百分比。
    - sem_percentage: 需要掩蔽的语义距离百分比。

    返回:
    - geo_mask: 地理掩码。
    - sem_mask: 语义掩码。
    - gnss_geo_mask: GNSS地理掩码。
    """
    es_geo_matrix = torch.tensor(es_geo_matrix.values.T, dtype=torch.float32)
    es_sem_matrix = torch.tensor(es_sem_matrix.values.T, dtype=torch.float32)
    gnss_geo_matrix = torch.tensor(gnss_geo_matrix.values.T, dtype=torch.float32)

    num_nodes = es_geo_matrix.shape[0]

    geo_threshold = torch.quantile(es_geo_matrix.flatten(), geo_percentage)
    geo_mask = es_geo_matrix > geo_threshold

    sem_threshold = torch.quantile(es_sem_matrix.flatten(), sem_percentage)
    sem_mask = es_sem_matrix > sem_threshold

    gnss_geo_threshold = torch.quantile(gnss_geo_matrix.flatten(), geo_percentage)
    gnss_geo_mask = gnss_geo_matrix > gnss_geo_threshold

    return geo_mask, sem_mask, gnss_geo_mask


class CombinedEarthquakeGNSSDataset(Dataset):
    def __init__(self, region_datasets):
        """
        用于组合不同区域数据集的类。

        参数：
        - region_datasets: 一个字典，键为区域名称，值为 EarthquakeGNSSDataset 的实例。
        """
        self.region_datasets = region_datasets
        self.region_names = list(region_datasets.keys())
        # 计算每个区域数据集的长度
        self.lengths = {region: len(ds) for region, ds in region_datasets.items()}
        # 计算总长度
        self.total_length = sum(self.lengths.values())
        self.lape_dim = region_datasets[self.region_names[0]].lape_dim
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        """
        根据全局索引检索一个样本。将全局索引映射到特定区域的数据集。
        """
        for region in self.region_names:
            if idx < self.lengths[region]:
                data = self.region_datasets[region][idx]
                data['region'] = region  # 在数据中包含区域名称，便于识别。
                return data
            idx -= self.lengths[region]
        raise IndexError("Index out of range in CombinedEarthquakeGNSSDataset.")

    def collate_fn(self, batch):
        """
        自定义的 collate 函数，用于处理具有不同 GNSS 节点数量的批次。

        参数：
        - batch: 来自不同区域的数据样本列表。
        返回：
        - 一个包含组合批次数据和区域特定掩码的字典。
        """
        # 动态计算此批次中 GNSS 节点的最大数量
        max_gnss_nodes_in_batch = max(item['gnss_data_history'].shape[1] for item in batch)
        # 初始化列表以存储填充后的数据和掩码
        log_energy_history_list = []
        gnss_data_history_list = []
        log_energy_future_list = []
        earthquake_data_future_day_list = []
        es_geo_masks = []
        es_sem_masks = []
        gnss_geo_masks = []
        lap_ex_list = []
        lap_gnss_list = []
        earthquake_happen_list = []
        earthquake_location_list = []
        station_location_list = []
        # 遍历批次中的每个样本，填充 GNSS 数据和掩码以匹配最大节点数
        gnss_padding_mask_list = []
        batch_copy = copy.deepcopy(batch)
        for item in batch_copy:
            gnss_padding_mask = torch.ones(max_gnss_nodes_in_batch, max_gnss_nodes_in_batch, dtype=torch.bool)
            num_gnss_nodes = item['gnss_data_history'].shape[1]
            pad_gnss_nodes = max_gnss_nodes_in_batch - num_gnss_nodes
            gnss_padding_mask[:num_gnss_nodes, :num_gnss_nodes] = False

            # 填充 gnss_data_history，在节点维度（第一个维度）进行填充
            padded_gnss_data = F.pad(
                item['gnss_data_history'],
                pad=(0, 0, 0, pad_gnss_nodes, 0, 0),  # (window_size, num_stations, num_features)
                mode='constant',
                value=0
            )
            
            # 追加数据
            log_energy_history_list.append(item['log_energy_history'])
            gnss_data_history_list.append(padded_gnss_data)
            log_energy_future_list.append(item['log_energy_future'])
            earthquake_data_future_day_list.append(item['earthquake_data_future_day'])

            # 获取区域特定的掩码
            region_name = item['region']

            # 填充 gnss_geo_mask
            gnss_geo_mask = item['gnss_geo_mask']

            padded_gnss_geo_mask = F.pad(
                gnss_geo_mask,
                pad=(0, pad_gnss_nodes, 0, pad_gnss_nodes),
                mode='constant',
                value=1  # 对于掩码，填充值为 1，表示连接不存在
            )

            # 填充 lap_gnss
            lap_gnss = item['lap_gnss']
            padded_lap_gnss = F.pad(
                lap_gnss,
                pad=(0, self.lape_dim - lap_gnss.shape[1], 0, pad_gnss_nodes),
                mode='constant',
                value=0
            )
            

            # 填充 es_geo_mask 和 es_sem_mask，使其形状一致
            es_geo_mask = item['es_geo_mask']
            es_sem_mask = item['es_sem_mask']
            earthquake_quake_location = item['earthquake_location']
            station_location = item['station_location']


            earthquake_quake_location, station_location = normalize_locations(earthquake_quake_location, station_location)

            padded_station_location = F.pad(
                station_location,
                pad=(0, 0, 0, pad_gnss_nodes),
                mode='constant',
                value=0
            )

            lap_ex = item['lap_ex']

            es_geo_masks.append(es_geo_mask)
            es_sem_masks.append(es_sem_mask)
            gnss_geo_masks.append(padded_gnss_geo_mask)
            lap_ex_list.append(lap_ex)
            lap_gnss_list.append(padded_lap_gnss)
            earthquake_happen_list.append(item['earthquake_happen'])
            gnss_padding_mask_list.append(gnss_padding_mask)
            earthquake_location_list.append(earthquake_quake_location)
            station_location_list.append(padded_station_location)
        
        log_energy_history = torch.stack(log_energy_history_list)
        gnss_data_history = torch.stack(gnss_data_history_list)
        log_energy_future = torch.stack(log_energy_future_list)
        earthquake_data_future_day = torch.stack(earthquake_data_future_day_list)

        es_geo_masks = torch.stack(es_geo_masks)
        es_sem_masks = torch.stack(es_sem_masks)
        gnss_geo_masks = torch.stack(gnss_geo_masks)
        lap_ex = torch.stack(lap_ex_list)
        lap_gnss = torch.stack(lap_gnss_list)
        earthquake_happen = torch.stack(earthquake_happen_list)
        gnss_padding_mask = torch.stack(gnss_padding_mask_list)
        earthquake_location = torch.stack(earthquake_location_list)
        station_location = torch.stack(station_location_list)

        # 组合成字典
        batch_data = {
            'log_energy_history': log_energy_history,
            'gnss_data_history': gnss_data_history,
            'log_energy_future': log_energy_future,
            'earthquake_data_future_day': earthquake_data_future_day,
            'es_geo_mask': es_geo_masks,
            'es_sem_mask': es_sem_masks,
            'gnss_geo_mask': gnss_geo_masks,
            'lap_ex': lap_ex,
            'lap_gnss': lap_gnss,
            'earthquake_happen':earthquake_happen,
            'gnss_padding_mask': gnss_padding_mask,
            'earthquake_location':earthquake_location,
            'station_location':station_location
        }

        return batch_data

def get_dataset(data_dir,window_size,forecast_horizon,lape_dim,geo_percentage,sem_percentage,
                time_resolution,earthquake_catalog_window):
    """
    Load the dataset from the specified directory.
    data_dir: Path to the directory containing the dataset files.
    window_size: Size of the historical window.
    forecast_horizon: Size of the future window.
    lape_dim: Number of dimensions for the graph Laplacian.
    far_mask_delta: Threshold for the far mask.
    dtw_delta: Threshold for the DT
    """
    area_list = os.listdir(data_dir)
    dataset_dict = {}
    for area in area_list:
        data_path = data_dir+area+"/"
        gnss_data = pd.read_csv(data_path + "gnss_data.csv", index_col=0, parse_dates=True, low_memory=False).map(parse_str_list)
        earthquake_data = pd.read_csv(data_path+"earthquake_data.csv", index_col=0, parse_dates=True)
        es_geo_matrix = pd.read_csv(data_path+"es_geo_matrix.csv", index_col=0)
        es_sem_matrix = pd.read_csv(data_path+"es_sem_matrix.csv", index_col=0)
        gnss_geo_matrix = pd.read_csv(data_path+"gnss_geo_matrix.csv", index_col=0)
        station_dict_use = pickle.load(open(data_path+"station_dict_use.pkl", "rb"))
        earthquake_dict_use = pickle.load(open(data_path+"grid_data/grid_id_map.pkl", "rb"))
        dataset_dict[area] =  EarthquakeGNSSDataset(area=area,
                                                    earthquake_data=earthquake_data,es_geo_matrix=es_geo_matrix,es_sem_matrix=es_sem_matrix,
                                                    gnss_geo_matrix=gnss_geo_matrix,gnss_data=gnss_data,geo_percentage=geo_percentage, sem_percentage=sem_percentage,
                                                    lape_dim=lape_dim,station_dict_use=station_dict_use,earthquake_dict_use=earthquake_dict_use,
                                                    window_size=window_size,forecast_horizon=forecast_horizon,earthquake_threshold=4,time_resolution=time_resolution,
                                                    earthquake_catalog_window = earthquake_catalog_window)
    dataset = CombinedEarthquakeGNSSDataset(dataset_dict)
    return dataset
