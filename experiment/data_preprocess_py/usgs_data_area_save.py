from datetime import datetime
import pandas as pd
from libcomcat.search import search
import os
import numpy as np
df_list=sorted(os.listdir("usgs_data_year"))

lat_min, lat_max = 32, 36
lon_min, lon_max = -120, -114
lat_step, lon_step = 0.1, 0.1

lat_bins = np.around(np.linspace(lat_min, lat_max, 41),2)
lon_bins = np.around(np.linspace(lon_min, lon_max, 61),2)

grid = {}

for lat in lat_bins[:-1]:
    for lon in lon_bins[:-1]:
        grid[(lat, lon)] = []

grid_id = {}

count = 0
for lat in lat_bins[:-1]:
    for lon in lon_bins[:-1]:
        grid_id[(lat, lon)] = count
        count += 1

for df_dir in df_list:
    df = pd.read_csv(os.path.join("usgs_data_year", df_dir))
    df['LatBin'] = pd.cut(df['Latitude'], bins=lat_bins, labels=lat_bins[:-1])
    df['LonBin'] = pd.cut(df['Longitude'], bins=lon_bins, labels=lon_bins[:-1])
    
    for _, row in df.iterrows():
        lat_bin = row['LatBin']
        lon_bin = row['LonBin']
        if pd.notna(lat_bin) and pd.notna(lon_bin):
            grid[(lat_bin, lon_bin)].append({
                'ID': row['ID'],
                'Time': row['Time'],
                'Magnitude': row['Magnitude'],
                'Latitude': row['Latitude'],
                'Longitude': row['Longitude'],
                'Depth': row['Depth'],
                'Location_id': grid_id[(lat_bin, lon_bin)]
            })
            #将df中的每个地震标上所在的网格编号
            df.loc[df['ID'] == row['ID'], 'Location_id'] = grid_id[(lat_bin, lon_bin)]
    df.to_csv(os.path.join("usgs_data_year", df_dir), index=False)

output_dir = 'grid_data'
os.makedirs(output_dir, exist_ok=True)

for key, events in grid.items():
    df_grid = pd.DataFrame(events)
    file_path = os.path.join(output_dir, f'grid_{np.around(key[0],2)}_{np.around(key[1],2)}.csv')
    df_grid.to_csv(file_path, index=False)

print("所有网格数据已保存。")