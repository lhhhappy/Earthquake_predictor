import pandas as pd
import pickle 
import os
import numpy as np
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

# 加载并排序网格ID
use_grid_id = pickle.load(open('use_grid_id.pkl', 'rb'))
use_grid_id.sort()

use_grid_id_dict = {use_grid_id[i]: i for i in range(len(use_grid_id))}

# 初始化全局Aij矩阵
def initialize_Aij():
    return np.zeros((len(use_grid_id), len(use_grid_id)))

# 更新连接矩阵，基于节点列表
def update_adjacency_matrix(A, node_list):
    for i, j in combinations(node_list, 2):
        A[i, j] += 1
        A[j, i] += 1  # 保证矩阵对称
    return A

# 根据第n天提取数据
def get_data_for_nth_day(df, n):
    return df[df['DayOfYear'] == n]

# 处理某一天的数据，更新Aij矩阵
def process_day(df, i, use_grid_id_dict, Aij):
    data = get_data_for_nth_day(df, i)
    earthquake_occur_id = data['Location_id'].fillna(-1).astype(int)
    # 避免同一天内同一个网格多次连边
    unique_grid_ids = [use_grid_id_dict[int(x)] for x in earthquake_occur_id if int(x) in use_grid_id_dict]
    unique_grid_ids = list(set(unique_grid_ids))  # 保证当天相同网格的地震只处理一次
    Aij = update_adjacency_matrix(Aij, unique_grid_ids)
    return Aij

# 处理单个年份文件
def process_single_file(df_dir):
    print(f"Processing file: {df_dir}")
    df = pd.read_csv(os.path.join("usgs_data_year", df_dir))
    df["Time"] = pd.to_datetime(df["Time"], format='ISO8601')
    df['DayOfYear'] = df['Time'].dt.dayofyear

    Aij = initialize_Aij()
    
    # 按天处理数据
    for i in range(1, 366):
        Aij = process_day(df, i, use_grid_id_dict, Aij)
    
    return Aij

# 多进程处理多个年份文件
def process_all_files_multithreaded(df_list):
    Aij_global = initialize_Aij()

    # 使用进程池处理每个文件
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, df_dir) for df_dir in df_list]

        # 串行累加每个年份处理完的Aij矩阵
        for future in futures:
            Aij = future.result()
            Aij_global += Aij  # 串行累加，避免竞争

    return Aij_global

# 获取df文件列表并排序
df_list = sorted(os.listdir("usgs_data_year"))

# 多线程处理所有文件
Aij_global = process_all_files_multithreaded(df_list)

# 保存Aij矩阵到文件
pickle.dump(Aij_global, open('Aij.pkl', 'wb'))

print("Processing complete. Aij matrix saved to 'Aij.pkl'.")