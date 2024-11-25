import torch
import torch.nn as nn   
import torch.nn.functional as F


class Mlpbaseline(nn.Module):
    def __init__(self, earthquake_dim, gnss_dim, embed_dim=64, 
                 earthquake_history_window=1400, gnss_history_window=140, 
                 gnss_station_num=30, earthquake_num=30,dropout=0.5):
        super(Mlpbaseline, self).__init__()
        
        # 定义处理地震数据的 MLP，每个地震事件独立处理
        self.earthquake_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(earthquake_history_window * earthquake_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 定义处理 GNSS 数据的 MLP，每个站点独立处理
        self.gnss_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gnss_history_window * 2 * gnss_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 定义输出层
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        self.gnss_station_num = gnss_station_num
        self.earthquake_num = earthquake_num
        
    def forward(self, earthquake, gnss_data, es_loc=None, gnss_loc=None, es_lap_mx=None, 
                gnss_lap_mx=None, es_geo_mask=None, es_sem_mask=None, 
                gnss_padding_mask=None, gnss_geo_mask=None):
        # 输入形状：
        # earthquake: [batch_size, earthquake_history_window, earthquake_num, earthquake_dim]
        # gnss_data: [batch_size, gnss_history_window, gnss_station_num, 2 * gnss_dim]
        
        batch_size = earthquake.size(0)
        
        # 处理地震数据，每个事件独立
        # 调整形状为 [batch_size * earthquake_num, earthquake_history_window, earthquake_dim]
        earthquake = earthquake.permute(0, 2, 1, 3).contiguous()
        earthquake = earthquake.view(-1, earthquake.size(2), earthquake.size(3))
        earthquake_features = self.earthquake_mlp(earthquake)
        
        # 处理 GNSS 数据，每个站点独立
        # 调整形状为 [batch_size * gnss_station_num, gnss_history_window, 2 * gnss_dim]
        gnss_data = gnss_data.permute(0, 2, 1, 3).contiguous()
        gnss_data = gnss_data.view(-1, gnss_data.size(2), gnss_data.size(3))
        gnss_features = self.gnss_mlp(gnss_data)
        
        # 重新调整形状回 batch_size
        earthquake_features = earthquake_features.view(batch_size, self.earthquake_num, -1)
        gnss_features = gnss_features.view(batch_size, self.gnss_station_num, -1)
        
        # 对 GNSS 特征进行聚合，例如取平均
        gnss_features = gnss_features.mean(dim=1, keepdim=True)
        gnss_features = gnss_features.expand(-1, self.earthquake_num, -1)
        
        # 将地震特征和 GNSS 特征拼接
        combined_features = torch.cat([earthquake_features, gnss_features], dim=-1)
        
        # 调整形状以输入到输出层
        combined_features = combined_features.view(-1, combined_features.size(-1))
        
        # 通过输出层
        output = self.output_mlp(combined_features)
        
        # 调整输出形状为 [batch_size, earthquake_num, 1]
        output = output.view(batch_size, self.earthquake_num, 1)
        
        return output, None
    
if __name__ == "__main__":
    batch_size = 2
    input_window = 1400
    num_nodes = 10
    feature_dim = 1
    embed_dim = 64
    lape_dim = 8
    gnss_station_num = 30
    gnss_dim = feature_dim * 2
    gnss_history_window = 140


    x = torch.randn(batch_size, input_window, num_nodes, feature_dim)
    gnss_data = torch.randn(batch_size, gnss_history_window,gnss_station_num, gnss_dim)
    es_lap_mx = torch.randn(batch_size,num_nodes, lape_dim)
    es_loc = torch.randn(batch_size,num_nodes, 2)
    gnss_loc = torch.randn(batch_size,gnss_station_num, 2)
    geo_mask = torch.zeros(batch_size, num_nodes, num_nodes).bool()
    sem_mask = torch.zeros(batch_size, num_nodes, num_nodes).bool()
    gnss_lap_mx = torch.randn(batch_size,gnss_station_num, lape_dim)
    gnss_geo_mask = torch.zeros(batch_size, gnss_station_num, gnss_station_num).bool()

    model = Mlpbaseline(earthquake_dim=feature_dim, gnss_dim=feature_dim, embed_dim=embed_dim, earthquake_history_window=input_window, gnss_history_window=gnss_history_window, gnss_station_num=gnss_station_num, earthquake_num=num_nodes)

    energy, day = model(earthquake=x, gnss_data=gnss_data, es_loc=es_loc, gnss_loc=gnss_loc, es_lap_mx=es_lap_mx, gnss_lap_mx=gnss_lap_mx, es_geo_mask=geo_mask, es_sem_mask=sem_mask, gnss_padding_mask=None,gnss_geo_mask=gnss_geo_mask)
    print(energy.shape)

