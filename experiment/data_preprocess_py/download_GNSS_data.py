import requests
from bs4 import BeautifulSoup
import zipfile
import os
import gzip
import pickle
import shutil
from bs4 import BeautifulSoup


station_dict_use = pickle.load(open('/data/linhang/workbench/Tipping_points_predictor/data/Southern California/station_dict/station_dict_use.pkl', 'rb'))
station_names = [i for i in station_dict_use.keys()]
base_url = "http://geodesy.unr.edu/NGLStationPages/stations/"


for station_name in station_names:
    url = "http://geodesy.unr.edu/gps_timeseries/kenv/" + station_name + "/"

    download_dir = "./downloads/"+station_name
    os.makedirs(download_dir, exist_ok=True)
    if os.listdir(download_dir):
        print(f"已经下载过: {station_name}")
        continue
    def download_file(file_url, download_path):
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            print(f"下载失败: {file_url}")

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    file_links = soup.find_all('a', href=True)
    for link in file_links:
        if link['href'].endswith('.zip'):
            file_url = url + link['href']
            file_name = link['href']
            download_path = os.path.join(download_dir, file_name)
            download_file(file_url, download_path)

# kenv_file_path = '/data/linhang/workbench/Tipping_points_predictor/downloads/'
# extracted_path = '/data/linhang/workbench/Tipping_points_predictor/data/'

# for file in station_names:
#     print('extracting', file)
#     os.makedirs(extracted_path + "/" + file, exist_ok=True)
#     current_env_file_path = kenv_file_path + file
#     zip_files = os.listdir(current_env_file_path)
#     for file_zip in zip_files:
#         if file_zip.endswith(".zip"): 
#             year = file_zip.split('.')[1]
#             output_file = extracted_path + "/" + file + "/" + year
#             os.makedirs(output_file, exist_ok=True)
#             zip_file = current_env_file_path + "/" + file_zip
            
#             try:
#                 with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#                     zip_ref.extractall(output_file)
#             except zipfile.BadZipFile:
#                 print(f"Error: {zip_file} is not a valid ZIP file.")
#         else:
#             print(f"Skipping non-ZIP file: {file_zip}")


# file_name=os.listdir(extracted_path)
# for file in file_name:
#     print(file)
#     kenv_file_path = extracted_path + file
#     kenv_file=os.listdir(kenv_file_path)
#     for kenv in kenv_file:
#         kenv_file_path_dir = kenv_file_path + "/" + kenv
#         all_file_dir=os.listdir(kenv_file_path_dir)
#         for file_dir in all_file_dir:
#             if not file_dir.endswith(".kenv.gz"):
#                 continue
#             end_file_dir=kenv_file_path_dir + "/" + file_dir
#             with gzip.open(end_file_dir, 'rb') as tar:
#                 with open(kenv_file_path_dir+ "/" + file_dir.split(".")[2]+".txt", 'wb') as out:
#                     shutil.copyfileobj(tar, out)
#             os.remove(end_file_dir)
