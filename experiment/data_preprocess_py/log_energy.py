from datetime import datetime
import pandas as pd
from libcomcat.search import search
import os
import numpy as np

data_list = os.listdir("data/grid_data")
os.makedirs('data/log_energy_data', exist_ok=True)
count = 0
use_grid_id = []
for data_dir in data_list:
    file_path = os.path.join("data/grid_data", data_dir)
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            data = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            continue
    if len(data) >= 86 : 
        data['Time'] = pd.to_datetime(data['Time'], format='ISO8601', errors='coerce')

        data.set_index('Time', inplace=True)
        data = data.dropna(subset=['Magnitude'])

        start_date = pd.Timestamp('1986-01-01', tz='UTC')
        end_date = pd.Timestamp('2024-08-31', tz='UTC')
        time_bins = pd.date_range(start=start_date, end=end_date, freq='2W')

        data['Time_bin'] = pd.cut(data.index, bins=time_bins, right=True)

        data['Time_bin'] = pd.cut(data.index, bins=time_bins, right=True)

        def calculate_energy(group):
            return np.sum(10 ** (1.5 * group['Magnitude']))

        grouped_energy = data.groupby('Time_bin',observed=False,).apply(lambda x: calculate_energy(x),include_groups=False)

        log_energy = ((1/1.5) * np.log10(grouped_energy.replace(0, np.nan))).replace(np.nan, 0)
        log_energy_filled = pd.DataFrame()
        log_energy_filled["Energy"] = log_energy.reindex(time_bins[:-1], fill_value=0)
        log_energy_filled["Location_id"] = data['Location_id'].iloc[0]
        log_energy_filled.index  = log_energy_filled.index.strftime('%Y-%m-%d')
        log_energy_filled.to_csv("data/log_energy_data/"+data_dir+".csv", index=True)
        count += 1

        use_grid_id.append(data['Location_id'].iloc[0])

print("所有数据已保存, 共有{}个网格数据".format(count))
import pickle
with open('use_grid_id.pkl', 'wb') as f:
    pickle.dump(use_grid_id, f)
        

