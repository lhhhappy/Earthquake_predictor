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
                 far_mask_delta, dtw_delta, lape_dim,
                 window_size=14, forecast_horizon=14, earthquake_threshold=4.0,
                 missing_threshold=5):
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

        # 计算数据集的长度
        self.length = len(earthquake_data) - window_size - forecast_horizon + 1

        # 生成掩码
        self.es_geo_mask, self.es_sem_mask, self.gnss_geo_mask = generate_masks(
            es_geo_matrix, es_sem_matrix, gnss_geo_matrix, far_mask_delta, dtw_delta
        )
        # 计算地震站点的图拉普拉斯嵌入
        self.lap_ex = graph_laplacian_embedding(torch.tensor(es_geo_matrix.values), lape_dim)

    def __len__(self):
        return self.length

    def __name__(self):
        return self.area

    def __getitem__(self, idx):
        # 获取历史和未来的地震数据
        earthquake_data_history = self.earthquake_data.iloc[idx:idx + self.window_size]
        earthquake_data_future = self.earthquake_data.iloc[
            idx + self.window_size:idx + self.window_size + self.forecast_horizon
        ]

        # 计算历史和未来的对数能量
        log_energy_history = calculate_energy_in_time_window(earthquake_data_history).values.T
        log_energy_future = calculate_energy_in_time_window(earthquake_data_future).values.T

        # 获取未来地震事件发生的天数
        earthquake_data_future_day = find_first_earthquake(
            earthquake_data_future, self.earthquake_threshold
        )

        # 获取GNSS数据的历史部分并处理
        gnss_data_history = self.gnss_data.iloc[idx:idx + self.window_size].values  # 形状：(window_size, num_stations)

        # 转置后，每个元素代表一个站点的时间序列数据
        gnss_data_history = gnss_data_history.T  # 形状：(num_stations, window_size)

        # 将每个站点的数据转换为固定长度的数组，形状：(num_stations, window_size, num_features)
        gnss_data_history = np.array([
            convert_to_fixed_length_array(station_data) for station_data in gnss_data_history
        ])

        # 检测并移除具有长连续缺失数据的站点
        missing_mask = np.isnan(gnss_data_history).all(axis=2)  # 形状：(num_stations, window_size)
        max_missing_lengths = self.max_consecutive_trues(missing_mask)
        stations_to_keep = max_missing_lengths <= self.missing_threshold

        # 更新gnss_data_history和gnss_geo_mask
        gnss_data_history = gnss_data_history[stations_to_keep]
        sample_gnss_geo_mask = self.gnss_geo_mask[stations_to_keep][:, stations_to_keep]

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

        sample_masks = {
            'es_geo_mask': self.es_geo_mask,
            'es_sem_mask': self.es_sem_mask,
            'gnss_geo_mask': sample_gnss_geo_mask,
            'lap_ex': self.lap_ex,
            'lap_gnss': sample_lap_gnss
        }

        return {
            'log_energy_history': log_energy_history,
            'gnss_data_history': gnss_data_history,
            'log_energy_future': log_energy_future,
            'earthquake_data_future_day': earthquake_data_future_day,
            'masks': sample_masks
        }

    def collate_fn(self, batch):
        batch_data = {}
        # 堆叠固定形状的数据
        batch_data['log_energy_history'] = torch.stack([item['log_energy_history'] for item in batch])
        batch_data['log_energy_future'] = torch.stack([item['log_energy_future'] for item in batch])
        batch_data['earthquake_data_future_day'] = torch.stack([item['earthquake_data_future_day'] for item in batch])

        # 处理可变数量的站点
        max_num_stations = max(item['gnss_data_history'].shape[0] for item in batch)
        gnss_data_histories = []
        gnss_geo_masks = []
        lap_gnss_list = []

        for item in batch:
            num_stations = item['gnss_data_history'].shape[0]
            # 计算需要填充的尺寸
            pad_stations = max_num_stations - num_stations

            # 填充gnss_data_history
            padded_data = F.pad(item['gnss_data_history'], (0, 0, 0, 0, 0, pad_stations), "constant", 0)
            gnss_data_histories.append(padded_data)

            # 填充gnss_geo_mask
            padded_mask = F.pad(item['masks']['gnss_geo_mask'], (0, pad_stations, 0, pad_stations), "constant", False)
            gnss_geo_masks.append(padded_mask)

            # 填充lap_gnss
            padded_lap = F.pad(item['masks']['lap_gnss'], (0, 0, 0, pad_stations), "constant", 0)
            lap_gnss_list.append(padded_lap)

        # 堆叠填充后的数据
        batch_data['gnss_data_history'] = torch.stack(gnss_data_histories)
        batch_data['masks'] = {
            'es_geo_mask': self.es_geo_mask,
            'es_sem_mask': self.es_sem_mask,
            'gnss_geo_mask': torch.stack(gnss_geo_masks),
            'lap_ex': self.lap_ex,
            'lap_gnss': torch.stack(lap_gnss_list)
        }

        return batch_data

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
    data_reshaped = data.reshape(num_stations * num_features, window_size)
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
    # 恢复原始形状
    data_filled = data_reshaped.reshape(num_stations, num_features, window_size)
    data_filled = data_filled.transpose(0, 2, 1)  # 形状：(num_stations, window_size, num_features)
    return data_filled

def generate_time_bins(start_date, end_date, freq=14):
    """
    高效批量生成以freq天为间隔的时间窗口。

    参数:
        start_date (str or pd.Timestamp): 开始日期。
        end_date (str or pd.Timestamp): 结束日期。
        freq (int): 间隔的天数。

    返回:
        pd.DatetimeIndex: 以freq天为间隔的时间窗口序列。
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    time_bins = pd.date_range(start=start_date, end=end_date + pd.Timedelta(days=freq), freq=f'{freq}D')
    return time_bins

def calculate_energy_in_time_window(data, freq=14):
    """
    计算在指定时间窗口内的能量。

    参数:
        data (pd.DataFrame): 包含站点数据的DataFrame，行名为日期，列名为站点名。
        freq (int): 时间窗口的间隔天数。

    返回:
        pd.DataFrame: 每个站点在每个时间窗口内的对数能量结果。
    """
    # 自动获取开始和结束日期
    start_date = data.index.min()
    end_date = data.index.max()

    time_bins = generate_time_bins(start_date, end_date, freq=freq)

    data = data.copy()
    # 将日期分配到时间窗口
    data['Time_bin'] = pd.cut(data.index, bins=time_bins, right=False)

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    # 过滤掉非正数值
    data[numeric_cols] = data[numeric_cols].where(data[numeric_cols] > 0)
    # 按时间窗口分组并求和
    grouped = data.groupby('Time_bin', observed=True)[numeric_cols].sum()
    # 计算能量
    grouped_energy = 10 ** (1.5 * grouped)
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
    - threshold (float): 震级阈值。

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

def generate_masks(es_geo_matrix, es_sem_matrix, gnss_geo_matrix, far_mask_delta, dtw_delta):
    """
    生成地理和语义掩码。

    参数:
    - es_geo_matrix: 地震-站点的地理邻接矩阵。
    - es_sem_matrix: 地震-站点的语义邻接矩阵。
    - gnss_geo_matrix: GNSS邻接矩阵。
    - far_mask_delta: 距离阈值。
    - dtw_delta: DTW阈值。

    返回:
    - geo_mask: 地理掩码。
    - sem_mask: 语义掩码。
    - gnss_geo_mask: GNSS地理掩码。
    """
    es_geo_matrix = torch.tensor(es_geo_matrix.values.T, dtype=torch.float32)
    es_sem_matrix = torch.tensor(es_sem_matrix.values.T, dtype=torch.float32)
    gnss_geo_matrix = torch.tensor(gnss_geo_matrix.values.T, dtype=torch.float32)

    num_nodes = es_geo_matrix.shape[0]
    gnss_station_num = gnss_geo_matrix.shape[0]

    # 生成geo_mask
    geo_mask = es_geo_matrix >= far_mask_delta
    # 生成sem_mask
    sem_mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
    sem_mask_indices = es_sem_matrix.argsort(dim=1)[:, :dtw_delta]
    sem_mask.scatter_(1, sem_mask_indices, False)
    # 生成gnss_geo_mask
    gnss_geo_mask = gnss_geo_matrix >= far_mask_delta

    return geo_mask, sem_mask, gnss_geo_mask
    

class CombinedEarthquakeGNSSDataset(Dataset):
    def __init__(self, region_datasets):
        """
        Class to combine datasets from different regions.
        
        :param region_datasets: Dictionary where keys are region names and values are instances of EarthquakeGNSSDataset.
        """
        self.region_datasets = region_datasets
        self.region_names = list(region_datasets.keys())
        self.lengths = {region: len(ds) for region, ds in region_datasets.items()}
        self.total_length = sum(self.lengths.values())
        self.max_gnss_nodes = max(ds[0]['gnss_data_history'].shape[1] for ds in region_datasets.values())

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        """
        Retrieve an item based on the global index. Maps the global index to a specific region dataset.
        """
        for region in self.region_names:
            if idx < self.lengths[region]:
                data = self.region_datasets[region][idx]
                data['region'] = region  # Include the region name in the data for identification.
                return data
            idx -= self.lengths[region]
        raise IndexError("Index out of range in CombinedEarthquakeGNSSDataset.")

    def collate_fn(self, batch):
        """
        Custom collate function to handle batches with different numbers of GNSS nodes.
        
        :param batch: List of data samples from different regions.
        :return: A dictionary containing combined batched data with region-specific masks.
        """
        # Determine the maximum number of GNSS nodes in this batch.
        max_gnss_nodes_in_batch = max(item['gnss_data_history'].shape[1] for item in batch)

        # Initialize lists to store the padded data and masks.
        log_energy_history = []
        gnss_data_history = []
        log_energy_future = []
        earthquake_data_future_day = []
        es_geo_masks = []
        es_sem_masks = []
        combined_gnss_masks = []
        lap_ex_masks = []
        lap_gnss_masks = []

        # Pad each data sample's GNSS data and masks to match the maximum number of nodes.
        for item in batch:
            n_gnss_nodes = item['gnss_data_history'].shape[1]
            pad_size = max_gnss_nodes_in_batch - n_gnss_nodes

            # Pad the GNSS data along the node dimension (dim=1)
            padded_gnss_data = F.pad(item['gnss_data_history'], (0, 0, 0, pad_size))
            # Append the data.
            log_energy_history.append(item['log_energy_history'])
            gnss_data_history.append(padded_gnss_data)
            log_energy_future.append(item['log_energy_future'])
            earthquake_data_future_day.append(item['earthquake_data_future_day'])

            # Retrieve region-specific masks.
            region_name = item['region']
            region_masks = self.region_datasets[region_name].get_masks()

            # Pad 'lap_gnss' and 'gnss_geo_mask' to match the maximum number of nodes.
            padded_gnss_geo_mask = F.pad(region_masks['gnss_geo_mask'], (0, pad_size,0, pad_size), value=1)

            padded_lap_gnss = F.pad(region_masks['lap_gnss'], (0, 0, 0, pad_size))
            # Combine GNSS padding mask with padded GNSS geo mask.
            gnss_padding_mask = torch.cat([torch.ones(n_gnss_nodes), torch.zeros(pad_size)], dim=0)
            combined_gnss_mask = gnss_padding_mask.unsqueeze(-1) * padded_gnss_geo_mask

            # Store the masks.
            es_geo_masks.append(region_masks['es_geo_mask'])
            es_sem_masks.append(region_masks['es_sem_mask'])
            combined_gnss_masks.append(combined_gnss_mask)
            lap_ex_masks.append(region_masks['lap_ex'])
            lap_gnss_masks.append(padded_lap_gnss)

        # Stack the data to create tensors for the batch.
        log_energy_history = torch.stack(log_energy_history)
        gnss_data_history = torch.stack(gnss_data_history)
        log_energy_future = torch.stack(log_energy_future)
        earthquake_data_future_day = torch.stack(earthquake_data_future_day)

        # Combine the padded masks.
        es_geo_masks = torch.stack(es_geo_masks)
        es_sem_masks = torch.stack(es_sem_masks)
        combined_gnss_masks = torch.stack(combined_gnss_masks)
        lap_ex_masks = torch.stack(lap_ex_masks)
        lap_gnss_masks = torch.stack(lap_gnss_masks)

        # Combine into a dictionary.
        batch_data = {
            'log_energy_history': log_energy_history,
            'gnss_data_history': gnss_data_history,
            'log_energy_future': log_energy_future,
            'earthquake_data_future_day': earthquake_data_future_day,
            'es_geo_mask': es_geo_masks,
            'es_sem_mask': es_sem_masks,
            'combined_gnss_mask': combined_gnss_masks,
            'lap_ex': lap_ex_masks,
            'lap_gnss': lap_gnss_masks
        }

        return batch_data

def get_dataset(data_dir,window_size,forecast_horizon,lape_dim,far_mask_delta,dtw_delta):
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
        gnss_data = pd.read_csv(data_path+"gnss_data.csv", index_col=0, parse_dates=True).map(parse_str_list)
        earthquake_data = pd.read_csv(data_path+"earthquake_data.csv", index_col=0, parse_dates=True)
        energy_data = pd.read_csv(data_path+"energy_data.csv", index_col=0, parse_dates=True)
        station_dict_use = pickle.load(open(data_path+"station_dict_use.pkl", "rb"))

        es_geo_matrix = pd.read_csv(data_path+"es_geo_matrix.csv", index_col=0)
        es_sem_matrix = pd.read_csv(data_path+"es_sem_matrix.csv", index_col=0)
        gnss_geo_matrix = pd.read_csv(data_path+"gnss_geo_matrix.csv", index_col=0)
        dataset_dict[area] =  EarthquakeGNSSDataset(area=area,
                                                    earthquake_data=earthquake_data,es_geo_matrix=es_geo_matrix,es_sem_matrix=es_sem_matrix,
                                                    gnss_geo_matrix=gnss_geo_matrix,gnss_data=gnss_data,far_mask_delta=far_mask_delta,
                                                    dtw_delta=dtw_delta,lape_dim=lape_dim,
                                                    window_size=window_size,forecast_horizon=forecast_horizon,earthquake_threshold=4)
    dataset = CombinedEarthquakeGNSSDataset(dataset_dict)
    return dataset
