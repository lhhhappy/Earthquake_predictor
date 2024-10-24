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

def parse_str_list(cell):
    try:
        return ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return cell
    
def get_gnss_i(series, index):
    #[n, e, u, vectors]
    return series.apply(lambda x: x[index] if isinstance(x, list) and len(x) > index else None)


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def fill_nan_with_interpolation(data):
    """
    对数据中的nan进行插值填充，使用矢量化操作进行优化。
    """
    # 创建数据的副本，避免修改原始数据
    data_filled = data.copy()

    # 处理每个维度上的NaN，避免双重循环
    for i in range(data_filled.shape[2]):  # 处理每个维度
        # 沿着第一个维度（行）对所有列应用插值函数
        def interpolate_row(row):
            mask = ~np.isnan(row)
            if np.sum(mask) > 1:
                # 创建插值函数
                interp_func = interp1d(np.where(mask)[0], row[mask], kind='linear', fill_value="extrapolate")
                return interp_func(np.arange(len(row)))
            return row  # 如果不能插值，则原样返回

        # 使用 np.apply_along_axis 对每一行（window_size维度）应用插值操作
        data_filled[:, :, i] = np.apply_along_axis(interpolate_row, axis=1, arr=data_filled[:, :, i])

    return data_filled


def convert_to_fixed_length_array(data, length=4):
    return [np.array(x[:length] if isinstance(x, list) else [np.nan]*length) for x in data]

def generate_time_bins(start_date, end_date, freq = 14):
    """
    高效批量生成从第一天开始的以2周为间隔的时间窗口。
    
    参数:
        start_date (str): 开始日期，格式 'YYYY-MM-DD'，例如 '2001-01-01'。
        end_date (str): 结束日期，格式 'YYYY-MM-DD'，例如 '2005-12-30'。
    
    返回:
        pd.DatetimeIndex: 以2周为间隔的时间窗口序列。
    """
    # 将输入的日期字符串转换为 Timestamp
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    total_days = (end_date - start_date).days
    num_intervals = total_days // freq

    time_bins = pd.DatetimeIndex([start_date + pd.Timedelta(days=14 * i) for i in range(num_intervals + 2)])

    return time_bins

def calculate_energy_in_time_window(data,freq=14):
    """
    计算在指定时间窗口内的能量。
    
    参数:
        data (pd.DataFrame): 包含站点数据的DataFrame，行名为日期，列名为站点名。
    
    输出:
        返回每个站点在每个时间窗口内的对数能量结果DataFrame。
    """
    # 自动获取开始和结束日期
    start_date = data.index.min()
    end_date = data.index.max()

    time_bins = generate_time_bins(start_date, end_date, freq=freq)

    data = data.copy()
    data['Time_bin'] = pd.cut(data.index, bins=time_bins, right=True)

    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].where(data[numeric_cols] > 0).dropna(how='all')

    grouped = data.groupby('Time_bin', observed=False).sum()
    grouped_energy = 10 ** (1.5 * grouped)

    log_energy = (1 / 1.5) * np.log10(grouped_energy.replace(0, np.nan))

    log_energy_filled = log_energy.fillna(0)

    log_energy_filled.index = log_energy_filled.index.categories.left.strftime('%Y-%m-%d')

    return log_energy_filled

def find_first_earthquake(earthquake_catalog, threshold):
    """
    优化后的版本，返回一个 (500, 1) 的向量，表示每个站点的第一个大于阈值的地震发生的天数（行数）。

    参数:
    - earthquake_catalog (pd.DataFrame): 地震目录，每个值为地震震级，行名为日期。
    - threshold (float): 震级阈值，筛选出大于此阈值的地震。

    返回:
    - result_vector (np.ndarray): 返回 (500, 1) 的向量，每个值表示第一个超过阈值的地震发生的行数（天数）。
    """
    earthquake_data = earthquake_catalog.to_numpy()

    above_threshold = earthquake_data > threshold

    result_vector = np.zeros((earthquake_data.shape[1], 1))

    first_event_indices = np.argmax(above_threshold, axis=0)

    has_event = above_threshold.any(axis=0)
    
    result_vector[has_event, 0] = first_event_indices[has_event] + 1

    return result_vector


def graph_laplacian_embedding(adj_matrix, k):
    """
    Compute the graph Laplacian embedding using the k smallest non-trivial eigenvectors.

    Parameters:
    adj_matrix (numpy.ndarray): Adjacency matrix of the graph.
    k (int): Number of smallest non-trivial eigenvectors to select.

    Returns:
    numpy.ndarray: Matrix of shape (N, k) representing the graph Laplacian embedding.
    """
    # Degree matrix
    D = np.diag(np.sum(adj_matrix, axis=1))
    
    # Compute D^(-1/2)
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    
    # Normalized Laplacian: I - D^(-1/2) * A * D^(-1/2)
    I = np.eye(adj_matrix.shape[0])
    laplacian = I - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = eigh(laplacian)
    
    # Select the k smallest non-trivial eigenvectors (skip the first one)
    # The first eigenvector corresponds to eigenvalue 0 (trivial solution).
    selected_eigenvectors = eigenvectors[:, 1:k+1]
    
    return torch.tensor(selected_eigenvectors, dtype=torch.float32)

def generate_masks(es_geo_matrix, es_sem_matrix, gnss_geo_matrix, far_mask_delta, dtw_delta):
    # Generate geo_mask
    es_geo_matrix = es_geo_matrix.T
    gnss_geo_matrix = gnss_geo_matrix.T

    es_geo_matrix = torch.tensor(es_geo_matrix.values, dtype=torch.float32)
    es_sem_matrix = torch.tensor(es_sem_matrix.values, dtype=torch.float32)
    gnss_geo_matrix = torch.tensor(gnss_geo_matrix.values, dtype=torch.float32)
    
    num_nodes = es_geo_matrix.shape[0]
    gnss_station_num = gnss_geo_matrix.shape[0]
    geo_mask = torch.zeros(num_nodes, num_nodes)
    geo_mask[(es_geo_matrix >= far_mask_delta)] = 1
    geo_mask = geo_mask.bool()
    
    # Generate sem_mask
    sem_mask = torch.ones(num_nodes, num_nodes)
    sem_mask_indices = es_sem_matrix.argsort(axis=1)[:, :dtw_delta]
    for i in range(sem_mask.shape[0]):
        sem_mask[i][sem_mask_indices[i]] = 0
    sem_mask = sem_mask.bool()
    
    # Generate gnss_geo_mask
    
    gnss_geo_mask = torch.zeros(gnss_station_num, gnss_station_num)
    gnss_geo_mask[gnss_geo_matrix >= far_mask_delta] = 1
    gnss_geo_mask = gnss_geo_mask.bool()
    
    return geo_mask, sem_mask, gnss_geo_mask

import torch
from torch.utils.data import Dataset
import numpy as np

class EarthquakeGNSSDataset(Dataset):
    def __init__(self, 
                 area,
                 earthquake_data, gnss_data,
                 es_geo_matrix, es_sem_matrix, gnss_geo_matrix,
                 far_mask_delta, dtw_delta, lape_dim,
                 window_size=14, forecast_horizon=14, earthquake_threshold=4.0):
        """
        Dataset class for the Earthquake-GNSS dataset.
        earthquake_data: DataFrame containing earthquake data.
        gnss_data: DataFrame containing GNSS data.
        window_size: Size of the historical window.
        forecast_horizon: Size of the future window.
        earthquake_threshold: Threshold for earthquake magnitude.
        es_geo_matrix: Earthquake-station adjacency matrix.
        es_sem_matrix: Earthquake-station semantic adjacency matrix.
        gnss_sem_matrix: GNSS semantic adjacency matrix.
        far_mask_delta: Threshold for the far mask.
        dtw_delta: Threshold for the DTW mask.
        lape_dim: Number of dimensions for the graph Laplacian.
        """
        self.earthquake_data = earthquake_data
        self.gnss_data = gnss_data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.earthquake_threshold = earthquake_threshold
        self.length = len(earthquake_data) - window_size - forecast_horizon + 1
        self.node_ = len(earthquake_data)
        self.area = area
        
        # Generate masks and store them in a dictionary
        self.masks = {
            'es_geo_mask': generate_masks(es_geo_matrix, es_sem_matrix, gnss_geo_matrix, far_mask_delta, dtw_delta)[0],
            'es_sem_mask': generate_masks(es_geo_matrix, es_sem_matrix, gnss_geo_matrix, far_mask_delta, dtw_delta)[1],
            'gnss_geo_mask': generate_masks(es_geo_matrix, es_sem_matrix, gnss_geo_matrix, far_mask_delta, dtw_delta)[2],
            'lap_ex': graph_laplacian_embedding(es_geo_matrix, lape_dim),
            'lap_gnss': graph_laplacian_embedding(gnss_geo_matrix, lape_dim),
        }
    
    def __len__(self):
        return self.length

    def __name__(self):
        return self.area
    
    def __getitem__(self, idx):
        # Get historical and future earthquake data
        earthquake_data_history = self.earthquake_data.iloc[idx:idx + self.window_size]
        earthquake_data_future = self.earthquake_data.iloc[idx + self.window_size:idx + self.window_size + self.forecast_horizon]

        # Calculate log energy history and future
        log_energy_history = np.array(calculate_energy_in_time_window(earthquake_data_history)).T
        log_energy_future = np.array(calculate_energy_in_time_window(earthquake_data_future)).T

        # Get the future earthquake data day threshold
        earthquake_data_future_day = find_first_earthquake(earthquake_data_future, self.earthquake_threshold)

        # Get the GNSS data history and process it
        gnss_data_history = self.gnss_data.iloc[idx:idx + self.window_size].values
        gnss_data_history = np.array([convert_to_fixed_length_array(row) for row in gnss_data_history.T])
        gnss_data_history = fill_nan_with_interpolation(gnss_data_history)

        # Convert data to tensors
        log_energy_history = torch.tensor(log_energy_history, dtype=torch.float32).permute(1, 0).unsqueeze(-1)
        log_energy_future = torch.tensor(log_energy_future, dtype=torch.float32)
        gnss_data_history = torch.tensor(gnss_data_history, dtype=torch.float32).permute(1, 0, 2)
        earthquake_data_future_day = torch.tensor(earthquake_data_future_day, dtype=torch.float32).squeeze(-1)

        return {
            'log_energy_history': log_energy_history,
            'gnss_data_history': gnss_data_history,
            'log_energy_future': log_energy_future,
            'earthquake_data_future_day': earthquake_data_future_day
        }
    
    def get_masks(self):
        """
        Retrieve the masks dictionary.
        :return: A dictionary containing all the masks.
        """
        return self.masks

    def collate_fn(self, batch):
        """
        Custom collate function to include masks in the batch.
        
        :param batch: List of data samples returned by the dataset's __getitem__ method.
        :param dataset: The dataset object that contains the masks.
        :return: A dictionary containing the batched data and the masks.
        """
        # Combine the batch data (list of dictionaries) into a single dictionary of tensors
        batch_data = {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
        
        # Add the masks to the batch data
        batch_data['masks'] = self.get_masks()
        
        return batch_data
    

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
