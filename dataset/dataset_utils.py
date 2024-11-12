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


def normalize_gnss(array_3d, method='min-max'):
    """
    对三维数组进行归一化，支持多种归一化方法。
    
    参数：
    - array_3d: 输入的三维数组，形状为 (时间步, 特征, 通道)。
    - method: 归一化方法，支持 'min-max', 'z-score', 'max-abs'。
    
    返回：
    - 归一化后的三维数组，形状与输入相同。
    """
    # 创建一个布尔掩码，标记有效的数据（非NaN的行）
    mask = ~np.isnan(array_3d).any(axis=2)  # (时间步, 特征)，标记有效行

    # 初始化一个与 array_3d 相同形状的归一化结果数组
    normalized_array = np.copy(array_3d)
    
    if method == 'min-max':
        # Min-Max 归一化，将值缩放到 [0, 1]
        min_vals = np.nanmin(np.where(mask[:, :, None], array_3d, np.nan), axis=0)  # (特征, 通道)
        max_vals = np.nanmax(np.where(mask[:, :, None], array_3d, np.nan), axis=0)  # (特征, 通道)
        
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # 避免分母为零

        normalized_array = (array_3d - min_vals) / range_vals
    
    elif method == 'z-score':
        # Z-score 标准化，使得均值为 0，标准差为 1
        mean_vals = np.nanmean(np.where(mask[:, :, None], array_3d, np.nan), axis=0)  # (特征, 通道)
        std_vals = np.nanstd(np.where(mask[:, :, None], array_3d, np.nan), axis=0)    # (特征, 通道)
        
        std_vals[std_vals == 0] = 1  # 避免分母为零

        normalized_array = (array_3d - mean_vals) / std_vals
    
    elif method == 'max-abs':
        # 最大绝对值归一化，将值缩放到 [-1, 1]
        max_abs_vals = np.nanmax(np.abs(np.where(mask[:, :, None], array_3d, np.nan)), axis=0)  # (特征, 通道)
        
        max_abs_vals[max_abs_vals == 0] = 1  # 避免分母为零

        normalized_array = array_3d / max_abs_vals
    
    else:
        raise ValueError("Unsupported normalization method. Choose from 'min-max', 'z-score', 'max-abs'.")

    normalized_array[~mask] = np.nan
    
    return normalized_array

def dataframe_to_array(df):
    df_filled = df.apply(lambda col: col.map(lambda x: x if isinstance(x, list) and len(x) == 4 else [np.nan] * 4))
    array_3d = np.array(df_filled.values.tolist()).reshape(df.shape[0], df.shape[1], 4)
    return array_3d

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
                 geo_percentage, sem_percentage, lape_dim,earthquake_dict_use,station_dict_use,
                 window_size=14, forecast_horizon=14, time_resolution = 14,
                 earthquake_threshold=4.0, missing_threshold=5):
        """
        地震-GNSS数据集的自定义Dataset类。

        参数：
        - area: 区域名称。
        - earthquake_data: 包含地震数据的DataFrame。
        - gnss_data: 包含GNSS数据的DataFrame。
        - es_geo_matrix: 地震站点的地理邻接矩阵。
        - es_sem_matrix: 地震站点的语义邻接矩阵。
        - gnss_geo_matrix: GNSS站点的地理邻接矩阵。
        - far_mask_delta: 远距离掩码的阈值。
        - dtw_delta: DTW掩码的阈值。
        - lape_dim: 图拉普拉斯嵌入的维度数。
        - window_size: 历史窗口大小，默认14。
        - forecast_horizon: 预测窗口大小，默认14。
        - earthquake_threshold: 地震震级阈值，默认4.0。
        - missing_threshold: 允许的最大连续缺失值数量，默认5。
        """
        self.area = area
        self.earthquake_data = earthquake_data
        self.gnss_data = gnss_data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.earthquake_threshold = earthquake_threshold
        self.missing_threshold = missing_threshold
        self.lape_dim = lape_dim
        self.time_resolution = time_resolution
        self.earthquake_loaction = torch.tensor([earthquake_dict_use[i] for i in sorted(earthquake_dict_use.keys())])
        self.station_loaction = np.array([station_dict_use[i] for i in sorted(station_dict_use.keys())],dtype=np.float32)


        self.normalized_gnss_data = dataframe_to_array(self.gnss_data)

        # 生成掩码
        self.es_geo_mask, self.es_sem_mask, self.gnss_geo_mask = generate_masks(
            es_geo_matrix, es_sem_matrix, gnss_geo_matrix, geo_percentage, sem_percentage
        )
        self.start_index = self.find_first_valid_index(self.normalized_gnss_data)
        
        self.length = len(earthquake_data) - window_size - forecast_horizon + 1 - self.start_index

        self.lap_ex = graph_laplacian_embedding(torch.tensor(es_geo_matrix.values), lape_dim)

    def __len__(self):
        return self.length

    def __name__(self):
        return self.area

    def __getitem__(self, idx):
        # 获取历史和未来的地震数据
        earthquake_data_history = self.earthquake_data.iloc[self.start_index+idx:self.start_index+idx + self.window_size]
        earthquake_data_future = self.earthquake_data.iloc[
            self.start_index+idx + self.window_size:idx + self.window_size + self.forecast_horizon+self.start_index
        ]

        earthquake_happen = torch.tensor((earthquake_data_future >= self.earthquake_threshold).any(axis=0).to_numpy(), dtype=torch.bool)

        # 计算历史和未来的对数能量
        log_energy_history = calculate_energy_in_time_window(earthquake_data_history,self.time_resolution).values.T
        log_energy_future = calculate_energy_in_time_window(earthquake_data_future,self.time_resolution).values.T

        # 获取未来地震事件发生的天数
        earthquake_data_future_day = find_first_earthquake(
            earthquake_data_future, self.earthquake_threshold
        )

        # 获取GNSS数据的历史部分并处理
        gnss_data_history = self.normalized_gnss_data[idx:idx + self.window_size].transpose(1,0,2)  # 形状：(window_size, num_stations)
        # 检测并移除具有长连续缺失数据的站点
        missing_mask = np.isnan(gnss_data_history).all(axis=2)
        max_missing_lengths = self.max_consecutive_trues(missing_mask)
        stations_to_keep = max_missing_lengths <= self.missing_threshold

        # 更新gnss_data_history和gnss_geo_mask
        gnss_data_history = gnss_data_history[stations_to_keep]
        station_location_use = self.station_loaction[stations_to_keep]

        sample_gnss_geo_mask = self.gnss_geo_mask[np.ix_(stations_to_keep, stations_to_keep)]

        # 生成GNSS数据的图拉普拉斯嵌入
        sample_lap_gnss = graph_laplacian_embedding(sample_gnss_geo_mask.float(), self.lape_dim)

        # 填充缺失值
        gnss_data_history = fill_nan_with_interpolation(gnss_data_history)

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
            'gnss_data_history': gnss_data_history.permute(1, 0, 2),
            'log_energy_future': log_energy_future,
            'earthquake_data_future_day': earthquake_data_future_day,
            'es_geo_mask': self.es_geo_mask,
            'es_sem_mask': self.es_sem_mask,
            'gnss_geo_mask': sample_gnss_geo_mask,
            'lap_ex': self.lap_ex,
            'lap_gnss': sample_lap_gnss,
            "earthquake_happen":earthquake_happen,
            "earthquake_loaction":self.earthquake_loaction,
            "station_loaction":station_location_use
        }
    def find_first_valid_index(self, gnss_data):
        """
        查找 GNSS 数据中第一个包含至少两个非 NaN 数据的时间点索引。
        """
        for idx, data in enumerate(gnss_data):
            # 统计非 NaN 元素数量
            if np.sum(~np.isnan(data)) >= 2:
                return idx
        return 0
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

import torch

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
        earthquake_loaction_list = []
        station_loaction_list = []
        # 遍历批次中的每个样本，填充 GNSS 数据和掩码以匹配最大节点数
        gnss_padding_mask_list = []
        for item in batch:

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
            region_dataset = self.region_datasets[region_name]

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
            earthquake_quake_loaction = item['earthquake_loaction']
            station_loaction = item['station_loaction']
            print(station_loaction.shape)
            padded_station_loaction = F.pad(
                station_loaction,
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
            earthquake_loaction_list.append(earthquake_quake_loaction)
            station_loaction_list.append(padded_station_loaction)
        
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
        earthquake_loaction = torch.stack(earthquake_loaction_list)
        station_loaction = torch.stack(station_loaction_list)

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
            'earthquake_loaction':earthquake_loaction,
            'station_loaction':station_loaction
        }

        return batch_data

def get_dataset(data_dir,window_size,forecast_horizon,lape_dim,geo_percentage,sem_percentage,time_resolution):
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
        dataset_dict[area] =  EarthquakeGNSSDataset(area=area,
                                                    earthquake_data=earthquake_data,es_geo_matrix=es_geo_matrix,es_sem_matrix=es_sem_matrix,
                                                    gnss_geo_matrix=gnss_geo_matrix,gnss_data=gnss_data,geo_percentage=geo_percentage, sem_percentage=sem_percentage,
                                                    lape_dim=lape_dim,
                                                    window_size=window_size,forecast_horizon=forecast_horizon,earthquake_threshold=4,time_resolution=time_resolution)
    dataset = CombinedEarthquakeGNSSDataset(dataset_dict)
    return dataset
