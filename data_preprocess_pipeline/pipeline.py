from utils import *
import pickle
import numpy as np
import pandas as pd

# 参数设置
earthquake_start_year = 1980
station_start_year = 2000
end_year = 2022

#latmin, latmax, lonmin, lonmax
#     
area_dict = {
    "Ibaraki": [35, 40, 139, 144],  # 茨城县
    "Hokkaido": [40, 45, 141, 147],  # 北海道
    "Miyagi": [37, 42, 140, 145],  # 宫城县
    "Fukushima": [36, 41, 140, 145],  # 福岛
    "Kumamoto": [32, 37, 130, 135],  # 熊本
    "Anchorage": [60, 65, -150, -145],  # 安克雷奇
    "California (Southern)": [32, 37, -120, -115],  # 加州南部
}

for area, [minlat, maxlat, minlon, maxlon] in area_dict.items():
    save_path = "/home/linhang/workbench/Earthquake_data/"+area
    topk = 100
    lat_step, lon_step = 0.1, 0.1

    # 时间范围
    
    station_start_stat = pd.to_datetime(f'{station_start_year}-01-01')
    earthquake_start_stat = pd.to_datetime(f'{earthquake_start_year}-01-01')
    end_date = pd.to_datetime(f'{end_year}-12-31')

    # 加载站点信息
    station_dict_all = pickle.load(open('data_preprocess_pipeline/station_dict_all.pkl', 'rb'))

    #修正 get_use_station_dict 函数定义中的参数顺序
    def get_use_station_dict(station_dict_all, minlatitude, maxlatitude, minlongitude, maxlongitude):
        station_dict_use = {}
        for station_name, station_dict in station_dict_all.items():
            lat, lon = eval(station_dict[0]), eval(station_dict[1])
            if minlatitude <= lat <= maxlatitude and minlongitude <= lon <= maxlongitude:
                station_dict_use[station_name] = station_dict
        return station_dict_use

    station_dict_use = get_use_station_dict(
        station_dict_all,
        minlatitude=minlat,
        maxlatitude=maxlat,
        minlongitude=minlon,
        maxlongitude=maxlon
    )
    station_names = list(station_dict_use.keys())
    print(f"处理区域: {area}")
    print(f"使用的站点数量: {len(station_names)}")
    if len(station_names) <= 30:
        print("该区域站点数量过少，跳过")
        continue

    os.makedirs(save_path, exist_ok=True)
    pickle.dump(station_dict_use, open(save_path+'/station_dict_use.pkl', 'wb'))

    # 下载地震数据和GNSS数据
    download_earthquake_data(earthquake_start_year, end_year, minlat, maxlat, minlon, maxlon, save_path+"/usgs_data_year",minmagnitude = 0)
    print("地震数据下载完成")
    download_GNSS_data(station_names, save_path+"/GNSS_day")
    print("GNSS数据下载完成")

    #处理USGS地震数据
    earthquake_dict = process_usgs_data(
            input_dir=save_path+"/usgs_data_year",
            output_dir=save_path+"/grid_data",
            lat_min=minlat,
            lat_max=maxlat,
            lon_min=minlon,
            lon_max=maxlon,
            lat_step=lat_step,
            lon_step=lon_step,
            topk=topk
        )
    print("地震数据处理完成")

    process_energy_data(
        input_dir=save_path+"/grid_data",
        output_dir=save_path+"/log_energy_data",
        start_date=earthquake_start_stat.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        freq='2W'
    )
    print("能量数据处理完成")
    # 构建地震数据CSV
    earthquake_csv = construct_earthquake_csv(
        directory=save_path+"/grid_data",
        start_date=earthquake_start_stat,
        end_date=end_date,
        output_file=save_path+'/earthquake_data.csv'
    )
    print("地震数据CSV构建完成")
    # 构建能量数据CSV
    construct_energy_csv(
        directory=save_path+'/log_energy_data',
        output_file=save_path+'/energy_data.csv'
    )
    print("能量数据CSV构建完成")
    # 构建GNSS数据CSV

    station_dict_fliter =  construct_gnss_csv(
                                directory=save_path+'/GNSS_day',
                                start_date=station_start_stat,
                                end_date=end_date,
                                output_file=save_path+'/gnss_data.csv',
                                station_dict=station_dict_use
                            )
    print("GNSS数据CSV构建完成")

    # 生成地震网格的邻接矩阵

    generate_sem_aij(
        data=earthquake_csv,
        output_file=save_path+'/es_sem_matrix.csv',
    )
    print("地震网格邻接矩阵生成完成")

    generate_geo_aij(
        station_dict=station_dict_fliter,
        save_path=save_path+'/gnss_geo_matrix.csv',
    )
    print("站点邻接矩阵生成完成")

    generate_geo_aij(
        station_dict=earthquake_dict,
        save_path=save_path+'/es_geo_matrix.csv'
    )