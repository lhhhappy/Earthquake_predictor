import torch
import torch.nn as nn   
import math
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial
from logging import getLogger
import os

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        B, T, N, C = x.size()

        front = x[:, 0:1, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
        end = x[:, -1:, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)

        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 3, 1).contiguous().view(B * N, C, -1)

        x = self.avg(x)

        x = x.view(B, N, C, -1).permute(0, 3, 1, 2).contiguous()
        return x
class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # Initialize with sinusoidal values
        pe = torch.zeros(max_len, embed_dim).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_dim)
        
        # Set as a learnable parameter
        self.pe = nn.Parameter(pe)

    def forward(self, x):
        # x: (batch_size, T, N, embed_dim)
        T = x.size(1)
        N = x.size(2)
        
        # Adjust positional encoding to match the input sequence length
        pe = self.pe[:, :T]  # (1, T, embed_dim)
        pe = pe.unsqueeze(2).expand(-1, -1, N, -1)  # (1, T, N, embed_dim)
        
        return pe

class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx)  # (batch_size, N, embed_dim)
        lap_pos_enc = lap_pos_enc.unsqueeze(1)  # (batch_size, 1, N, embed_dim)
        return lap_pos_enc
    
class LocationEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.location_embed = nn.Linear(input_dim, embed_dim)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.location_embed(x)
        x = self.norm(x).unsqueeze(1)
        return x
    
class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, lape_dim, drop=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)

        self.position_encoding = PositionalEmbedding(embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)
        self.location_embedding = LocationEmbedding(2, embed_dim)

    def forward(self, x, lap_mx, loc):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)
        x += self.spatial_embedding(lap_mx)
        x += self.location_embedding(loc)
        x = self.dropout(x)
        return x
    
class MultiResolutionTimeDownsampling(nn.Module):
    def __init__(self, down_sampling_method='avg', down_sampling_window=2, down_sampling_layers=3):
        super(MultiResolutionTimeDownsampling, self).__init__()
        
        self.down_sampling_layers = nn.ModuleList()
        
        for _ in range(down_sampling_layers):
            if down_sampling_method == 'max':
                layer = nn.MaxPool1d(down_sampling_window, stride=down_sampling_window)
            elif down_sampling_method == 'avg':
                layer = nn.AvgPool1d(down_sampling_window, stride=down_sampling_window)
            elif down_sampling_method == 'conv':
                layer = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=down_sampling_window, stride=down_sampling_window, padding=0)
            else:
                raise ValueError("Unsupported down_sampling_method. Choose from 'max', 'avg', or 'conv'.")
                
            self.down_sampling_layers.append(layer)
        
    def forward(self, x):
        """
        前向计算函数，执行时间维度的多层下采样
        :param x: 输入张量，形状为 (B, T, N, C)
        :return: 下采样后的张量列表，每一层的结果均保存在列表中
        """
        x = x.permute(0, 3, 2, 1).contiguous()  # 转换为 (B, C, N, T)
        B, C, N, T = x.size()
        outputs = []
        x_in = x  # 保留原始输入
        outputs.append(x_in.permute(0, 3, 2, 1).contiguous())
        for layer in self.down_sampling_layers:
            # 将 x_in 进行下采样
            x_flat = x_in.view(B * C * N, 1, T)  # (B*C*N, 1, T)
            x_downsampled = layer(x_flat)  # 下采样结果，形状为 (B*C*N, 1, T_new)
            x_downsampled = x_downsampled.squeeze(1)  # 去除通道维度，(B*C*N, T_new)
            T_new = x_downsampled.size(-1)
            # 恢复形状
            x_downsampled = x_downsampled.view(B, C, N, T_new)
            # 保存下采样结果
            outputs.append(x_downsampled.permute(0, 3, 2, 1).contiguous())  # 转换回 (B, T_new, N, C)
            # 更新 x_in 和 T
            x_in = x_downsampled
            T = T_new

        return outputs


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, seq_len, down_sampling_window, d_model, d_ff, down_sampling_layers):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers = nn.ModuleList()

        for i in range(down_sampling_layers):
            input_dim = seq_len // (down_sampling_window ** i)
            output_dim = seq_len // (down_sampling_window ** (i + 1))
            self.down_sampling_layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.GELU(),
                    nn.Linear(output_dim, output_dim),
                )
            )

    def forward(self, season_list):
        # season_list 中的每个元素形状为 [B, C, N, T]
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high]

        for i in range(len(season_list) - 1):
            B, C, N, T = out_high.size()
            # 调整维度为 [B * C * N, T]，以在时间维度上应用线性层
            out_high_reshaped = out_high.permute(0, 2, 1, 3).contiguous().view(B * N * C, T)
            # 通过线性层下采样时间维度
            out_low_res = self.down_sampling_layers[i](out_high_reshaped)  # [B * N * C, T_new]
            T_new = out_low_res.size(1)
            # 恢复形状为 [B, C, N, T_new]
            out_low_res = out_low_res.view(B, N, C, T_new).permute(0, 2, 1, 3).contiguous()
            # 更新 out_low 和 out_high
            out_low = out_low + out_low_res
            out_high = out_low

            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]

            out_season_list.append(out_high)
        out_season_list = [x.permute(0, 3, 2, 1).contiguous() for x in out_season_list]
        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, seq_len, down_sampling_window, d_model, d_ff, down_sampling_layers):
        super(MultiScaleTrendMixing, self).__init__()
        self.up_sampling_layers = nn.ModuleList()

        for i in reversed(range(down_sampling_layers)):
            input_dim = seq_len // (down_sampling_window ** (i + 1))
            output_dim = seq_len // (down_sampling_window ** i)
            self.up_sampling_layers.append(
                nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.GELU(),
                    nn.Linear(output_dim, output_dim),
                )
            )

    def forward(self, trend_list):
        # trend_list 中的每个元素形状为 [B, C, N, T]
        trend_list_reverse = trend_list[::-1]
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low]

        for i in range(len(trend_list_reverse) - 1):
            B, C, N, T = out_low.size()
            # 调整维度为 [B * N * C, T]
            out_low_reshaped = out_low.permute(0, 2, 1, 3).contiguous().view(B * N * C, T)
            # 通过线性层上采样时间维度
            out_high_res = self.up_sampling_layers[i](out_low_reshaped)  # [B * N * C, T_new]
            T_new = out_high_res.size(1)
            # 恢复形状为 [B, C, N, T_new]
            out_high_res = out_high_res.view(B, N, C, T_new).permute(0, 2, 1, 3).contiguous()
            # 更新 out_high 和 out_low
            out_high = out_high + out_high_res
            out_low = out_high

            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]

            out_trend_list.append(out_low)

        out_trend_list.reverse()
        out_trend_list = [x.permute(0, 3, 2, 1).contiguous() for x in out_trend_list]
        return out_trend_list
    
class PastDecomposableMixing(nn.Module):
    def __init__(self, seq_len, down_sampling_window, d_model, d_ff, dropout, decomp_method='moving_avg', 
                 moving_avg_kernel=3, top_k=5, down_sampling_layers=3):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # Initialize decomposition method
        if decomp_method == 'moving_avg':
            self.decomposition = series_decomp(moving_avg_kernel)
        else:
            raise ValueError("Invalid decomposition method")

        # Cross layer only used if channel independence is disabled
        self.cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

        # Multi-scale mixing modules for season and trend
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(seq_len, down_sampling_window, d_model, d_ff, down_sampling_layers)
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(seq_len, down_sampling_window, d_model, d_ff, down_sampling_layers)

        # Output cross-layer
        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x_list):
        length_list = [x.size(1) for x in x_list]  # Extract the time length of each x in x_list

        # Decompose each scale of x_list to obtain season and trend components
        season_list = []
        trend_list = []
        for x in x_list:
            B, T, N, C = x.size()
            x = x.permute(0, 3, 2, 1).contiguous()  # Change to [B, C, N, T] for decomposition

            season, trend = self.decomposition(x)  # Decompose x to get season and trend components

            season = self.cross_layer(season.permute(0, 3, 2, 1))  # Restore to [B, T, N, C]
            trend = self.cross_layer(trend.permute(0, 3, 2, 1))  # Same adjustment

            season_list.append(season.permute(0, 3, 2, 1))  # Convert back to [B, T, N, C]
            trend_list.append(trend.permute(0, 3, 2, 1))  # Same

        # Bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # Top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        # Combine season and trend, apply cross-layer processing if needed
        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :, :])  # Truncate to the original length

        return out_list

class NodeSelfAttention(nn.Module):
    def __init__(
        self, dim, geo_num_heads=4, sem_num_heads=2, qkv_bias=False,
        attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % (geo_num_heads + sem_num_heads) == 0
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.head_dim = dim // (geo_num_heads + sem_num_heads)
        self.scale = self.head_dim ** -0.5
        self.geo_ratio = geo_num_heads / (geo_num_heads + sem_num_heads) if geo_num_heads > 0 else 0
        self.sem_ratio = 1 - self.geo_ratio

        # Conditional initialization based on the number of heads
        self.geo_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias) if geo_num_heads > 0 else None
        self.geo_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias) if geo_num_heads > 0 else None
        self.geo_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias) if geo_num_heads > 0 else None
        self.geo_attn_drop = nn.Dropout(attn_drop) if geo_num_heads > 0 else None

        self.sem_q_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias) if sem_num_heads > 0 else None
        self.sem_k_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias) if sem_num_heads > 0 else None
        self.sem_v_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias) if sem_num_heads > 0 else None
        self.sem_attn_drop = nn.Dropout(attn_drop) if sem_num_heads > 0 else None

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.layernorm = nn.LayerNorm(dim)
    def forward(self, x, geo_mask=None, sem_mask=None, padding_mask=None):
        B, T, N, D = x.shape
        original_x = x
        if self.geo_num_heads > 0:
            geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            geo_k = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            geo_v = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale  # Shape: (B, T, geo_num_heads, N, N)
            if geo_mask is not None:
                geo_mask = geo_mask.unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, N, N)
                geo_mask = geo_mask.bool()
                geo_attn = geo_attn.masked_fill(geo_mask, float('-inf'))
            geo_attn = geo_attn.softmax(dim=-1)
            if padding_mask is not None:
                padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
                padding_mask = padding_mask.bool()
                geo_attn = geo_attn.masked_fill(padding_mask, 0)

            geo_attn = self.geo_attn_drop(geo_attn)
            geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, int(D * self.geo_ratio))
        else:
            geo_x = torch.zeros(B, T, N, int(D * self.geo_ratio), device=x.device)

        if self.sem_num_heads > 0:
            sem_q = self.sem_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            sem_k = self.sem_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            sem_v = self.sem_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            sem_q = sem_q.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            sem_k = sem_k.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            sem_v = sem_v.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
            if sem_mask is not None:
                # Adjust sem_mask to shape (B, 1, 1, N, N) to broadcast over T and sem_num_heads
                sem_mask = sem_mask.unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, N, N)
                sem_mask = sem_mask.bool()
                sem_attn = sem_attn.masked_fill(sem_mask, float('-inf'))
            sem_attn = sem_attn.softmax(dim=-1)
            if padding_mask is not None:
                padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
                padding_mask = padding_mask.bool()
                sem_attn = sem_attn.masked_fill(padding_mask, 0)
            sem_attn = self.sem_attn_drop(sem_attn)
            sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, int(D * self.sem_ratio))
        else:
            sem_x = torch.zeros(B, T, N, int(D * self.sem_ratio), device=x.device)

        x = self.proj(torch.cat([geo_x, sem_x], dim=-1))
        x = self.proj_drop(x)
        x = self.layernorm(original_x + x)
        return x
class NodeAttentionBlock(nn.Module):
    def __init__(self, dim, geo_num_heads=4, sem_num_heads=2, qkv_bias=False,
                 attn_drop=0., proj_drop=0.,down_sampling_layers=2):
        super().__init__()
        self.nodeattn = nn.ModuleList([
            NodeSelfAttention(dim=dim, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=proj_drop) for _ in range(down_sampling_layers+1)
        ])
    def forward(self, x_list, geo_mask=None, sem_mask=None, padding_mask=None):
        out_list = []
        for x, attn in zip(x_list, self.nodeattn):
            out_list.append(attn(x, geo_mask, sem_mask, padding_mask))
        return out_list
class TemporalConvCompression(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(TemporalConvCompression, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=output_dim, 
            kernel_size=kernel_size, 
            stride=1,
            padding=kernel_size // 2
        )
        # 将时间维度缩减到1
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [B, T, N, C]
        B, T, N, C = x.size()
        # 调整形状以适应 Conv1d，合并批次和节点维度
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, N, C, T]
        x = x.view(B * N, C, T)  # [B*N, C, T]
        x = self.conv(x)  # [B*N, output_dim, T]
        x = self.activation(x)
        x = self.pool(x)  # [B*N, output_dim, 1]
        x = x.squeeze(-1)  # [B*N, output_dim]
        x = x.view(B, N, -1)  # [B, N, output_dim]
        return x

class MultiScaleFusionModule(nn.Module):
    def __init__(self, input_dim, num_scales, compress_dims, fusion_dim, output_dim, kernel_size=3):
        super(MultiScaleFusionModule, self).__init__()
        # 参数说明：
        # input_dim: 输入特征维度 C
        # num_scales: 多尺度数量（输入列表的长度）
        # compress_dims: 每个尺度压缩后的特征维度列表
        # fusion_dim: 融合后的特征维度
        # output_dim: 最终输出特征维度 D_model
        # kernel_size: 时间卷积的卷积核大小
        
        # 创建每个尺度的时间卷积压缩层
        self.temporal_compression_layers = nn.ModuleList([
            TemporalConvCompression(
                input_dim=input_dim,
                output_dim=compress_dims[i],
                kernel_size=kernel_size
            )
            for i in range(num_scales)
        ])

        self.fusion_layer = nn.Linear(compress_dims[0], fusion_dim)
        self.output_layer = nn.Linear(fusion_dim, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x_list):
        multiscale_features = 0
        for i, x in enumerate(x_list):
            compressed_feature = self.temporal_compression_layers[i](x)  # [B, N, compress_dim_i]
            multiscale_features += compressed_feature
            
        fused_feature = self.fusion_layer(multiscale_features)  # [B, N, fusion_dim]
        fused_feature = self.activation(fused_feature)
        output = self.output_layer(fused_feature)  # [B, N, output_dim]
        
        return output
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class STSelfAttention(nn.Module):
    def __init__(
        self, dim, geo_num_heads=4,t_num_heads=2, qkv_bias=False,
        attn_drop=0., proj_drop=0., output_dim=1):
        super().__init__()
        assert dim % (geo_num_heads + t_num_heads) == 0
        self.geo_num_heads = geo_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (geo_num_heads + t_num_heads)
        self.scale = self.head_dim ** -0.5
        self.geo_ratio = geo_num_heads / (geo_num_heads + t_num_heads) if geo_num_heads > 0 else 0
        self.t_ratio = 1 - self.geo_ratio
        self.output_dim = output_dim

        # Conditional initialization based on the number of heads
        self.geo_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias) if geo_num_heads > 0 else None
        self.geo_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias) if geo_num_heads > 0 else None
        self.geo_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias) if geo_num_heads > 0 else None
        self.geo_attn_drop = nn.Dropout(attn_drop) if geo_num_heads > 0 else None

        self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, geo_mask=None, padding_mask=None):
        B, T, N, D = x.shape

        # Temporal block
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, int(D * self.t_ratio)).transpose(1, 2)

        # Geographic block
        if self.geo_num_heads > 0:
            geo_q = self.geo_q_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            geo_k = self.geo_k_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            geo_v = self.geo_v_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            geo_k = geo_k.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            geo_v = geo_v.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale  # Shape: (B, T, geo_num_heads, N, N)
            if geo_mask is not None:
                # Adjust geo_mask to shape (B, 1, 1, N, N) to broadcast over T and geo_num_heads
                geo_mask = geo_mask.unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, N, N)
                geo_mask = geo_mask.bool()
                geo_attn = geo_attn.masked_fill(geo_mask, float('-inf'))
            geo_attn = geo_attn.softmax(dim=-1)
            if padding_mask is not None:
                # Adjust padding_mask to shape (B, 1, 1, N, N) to broadcast over T and geo_num_heads
                padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
                padding_mask = padding_mask.bool()
                geo_attn = geo_attn.masked_fill(padding_mask, 0)

            geo_attn = self.geo_attn_drop(geo_attn)
            geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, int(D * self.geo_ratio))
        else:
            geo_x = torch.zeros(B, T, N, int(D * self.geo_ratio), device=x.device)

        # Concatenate and project output
        x = self.proj(torch.cat([t_x, geo_x], dim=-1))
        x = self.proj_drop(x)
        return x

class STEncoderBlock(nn.Module):
    
    def __init__(
        self, dim, geo_num_heads=4, t_num_heads=2, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, type_ln="pre",
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(
            dim, geo_num_heads=geo_num_heads, t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, geo_mask=None,padding_mask = None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(self.st_attn(self.norm1(x), geo_mask=geo_mask, padding_mask = padding_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(x + self.drop_path(self.st_attn(x, geo_mask=geo_mask, padding_mask = padding_mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x

class CustomCrossAttentionNetwork(torch.nn.Module):
    def __init__(self, d_model):
        super(CustomCrossAttentionNetwork, self).__init__()
        # 1D convolutions to compute q, k, v embeddings
        self.q_conv = torch.nn.Conv1d(d_model, d_model, kernel_size=1)
        self.k_conv = torch.nn.Conv1d(d_model, d_model, kernel_size=1)
        self.v_conv = torch.nn.Conv1d(d_model, d_model, kernel_size=1)
        self.scale = d_model ** -0.5

        # LayerNorm for enc1 and enc2
        self.layernorm_enc1 = torch.nn.LayerNorm(d_model)
        self.layernorm_enc2 = torch.nn.LayerNorm(d_model)

        # Final LayerNorm for output
        self.layernorm_out = torch.nn.LayerNorm(d_model)

    def forward(self, enc1, enc2):
        # enc1: (B, N1, d_model), enc2: (B, N2, d_model)
        B, N1, d_model = enc1.size()
        B, N2, _ = enc2.size()

        # Apply LayerNorm to enc1 and enc2
        enc1 = self.layernorm_enc1(enc1)  # (B, N1, d_model)
        enc2 = self.layernorm_enc2(enc2)  # (B, N2, d_model)

        # Reshape for conv1d: (B, d_model, N)
        enc1 = enc1.permute(0, 2, 1)  # (B, d_model, N1)
        enc2 = enc2.permute(0, 2, 1)  # (B, d_model, N2)

        # Apply convolutions to obtain q, k, v
        q = self.q_conv(enc1)  # (B, d_model, N1)
        k = self.k_conv(enc2)  # (B, d_model, N2)
        v = self.v_conv(enc2)  # (B, d_model, N2)

        # Reshape q, k, v for dot-product attention: (B, N, d_model)
        q = q.permute(0, 2, 1)  # (B, N1, d_model)
        k = k.permute(0, 2, 1)  # (B, N2, d_model)
        v = v.permute(0, 2, 1)  # (B, N2, d_model)

        # Compute attention scores and apply scaling
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, N1, N2)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, N1, N2)

        # Weighted sum of values based on attention weights
        attn_output = torch.bmm(attn_weights, v)  # (B, N1, d_model)

        # Add residual connection and apply LayerNorm to the output
        enc1_residual = enc1.permute(0, 2, 1)  # Convert back to (B, N1, d_model)

        # Combine attn_output and residual, then normalize
        output = self.layernorm_out(attn_output + enc1_residual)
        return output
    


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features  # Number of feature channels (C)
        self.eps = eps
        self.affine = affine
        if self.affine:
            # Parameters for scaling and shifting
            self.gamma = nn.Parameter(torch.ones(1, 1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x, mode):
        if mode == 'norm':
            # Compute mean and std over T for each (B, N, C)
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = x.std(dim=1, keepdim=True, unbiased=False).detach()
            # Normalize
            x = (x - self.mean) / (self.std + self.eps)
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        elif mode == 'denorm':
            # Denormalize
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * (self.std + self.eps) + self.mean
            return x
        else:
            raise ValueError("Mode must be 'norm' or 'denorm'")
    

class ES_net_mixer(nn.Module):
    def __init__(self, earthquake_dim, gnss_dim, embed_dim=64, lape_dim=8, earthquake_history_window=1400,
                 down_sampling_method='avg', down_sampling_window=2, down_sampling_layers=3, 
                 pdm_layers=2, pdm_d_model=64, pdm_d_ff=128, 
                 pdm_dropout=0.1, pdm_decomp_method='moving_avg', pdm_moving_avg_kernel=3, 
                 geo_num_heads=4, sem_num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 mlp_ratio=4., enc_depth=2,type_ln="pre", prediction_day_head=1 , prediction_energy_len = 1,
                 use_rev_in=False):
        super(ES_net_mixer, self).__init__()
        
        self.earthquake_dim = earthquake_dim
        self.gnss_dim = gnss_dim
        self.embed_dim = embed_dim
        self.lape_dim = lape_dim
        self.down_sampling_method = down_sampling_method
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        self.pdm_layers = pdm_layers
        self.pdm_seq_len = earthquake_history_window
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.mlp_ratio = mlp_ratio
        self.type_ln = type_ln
        self.enc_depth = enc_depth
        self.pdm_d_model = pdm_d_model
        self.pdm_d_ff = pdm_d_ff
        self.pdm_dropout = pdm_dropout
        self.pdm_decomp_method = pdm_decomp_method
        self.pdm_moving_avg_kernel = pdm_moving_avg_kernel
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.prediction_day_head = prediction_day_head
        self.predict_energy_len = prediction_energy_len

        # Embeddings for earthquake and GNSS data
        self.earthquake_embedding = DataEmbedding(
            feature_dim=self.earthquake_dim, embed_dim=self.embed_dim, lape_dim=self.lape_dim)
        
        self.gnss_embedding_trend = DataEmbedding(
            feature_dim=self.gnss_dim, embed_dim=self.embed_dim, lape_dim=self.lape_dim)
        
        self.gnss_embedding_season = DataEmbedding(
            feature_dim=self.gnss_dim, embed_dim=self.embed_dim, lape_dim=self.lape_dim)
        
        # Multi-resolution downsampling
        self.multi_resolution_time_downsampling = MultiResolutionTimeDownsampling(
            down_sampling_method=self.down_sampling_method, 
            down_sampling_window=self.down_sampling_window,
            down_sampling_layers=self.down_sampling_layers
        )

        # PastDecomposableMixing layers
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                seq_len=self.pdm_seq_len, 
                down_sampling_window=self.down_sampling_window, 
                d_model=self.pdm_d_model, 
                d_ff=self.pdm_d_ff, 
                dropout=self.pdm_dropout, 
                decomp_method=self.pdm_decomp_method, 
                moving_avg_kernel=self.pdm_moving_avg_kernel, 
            ) 
            for _ in range(self.pdm_layers)
        ])

        self.node_self_attention_block = nn.ModuleList([
            NodeAttentionBlock(dim=pdm_d_model, geo_num_heads=self.geo_num_heads, sem_num_heads=self.sem_num_heads, qkv_bias=self.qkv_bias,
                               attn_drop=self.attn_drop, proj_drop=self.proj_drop, down_sampling_layers=down_sampling_layers)
            for _ in range(self.pdm_layers)
        ])

        self.multi_scale_fusion_module = MultiScaleFusionModule(
            input_dim=self.pdm_d_model, 
            num_scales=self.down_sampling_layers + 1, 
            compress_dims=[self.pdm_d_model] * (self.down_sampling_layers + 1), 
            fusion_dim=self.pdm_d_model, 
            output_dim=self.pdm_d_model
        )

        self.encoder_blocks_trend = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim,
                geo_num_heads=self.geo_num_heads, t_num_heads=self.sem_num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.attn_drop, 
                attn_drop=self.attn_drop, act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), type_ln=self.type_ln,
            ) for i in range(enc_depth)
        ])
        self.skip_convs_trend = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.pdm_d_model, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

        self.encoder_blocks_season = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim,
                geo_num_heads=self.geo_num_heads, t_num_heads=self.sem_num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.attn_drop, 
                attn_drop=self.attn_drop, act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), type_ln=self.type_ln,
            ) for i in range(enc_depth)
        ])
        self.skip_convs_season = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.pdm_d_model, kernel_size=1,
            ) for _ in range(enc_depth)
        ])
        self.layernorm = nn.LayerNorm(self.pdm_d_model)
        
        self.gnss_compression = TemporalConvCompression(
            input_dim=self.pdm_d_model, output_dim=self.embed_dim, kernel_size=3
        )
        
        self.cross_attention_network = CustomCrossAttentionNetwork(d_model=self.pdm_d_model)
        
        if self.prediction_day_head != 0:
            self.predict_day_head = Mlp(in_features=self.pdm_d_model, hidden_features=self.pdm_d_model, 
                                        out_features=self.prediction_day_head)
            
        self.predict_energy_head = Mlp(in_features=self.pdm_d_model, hidden_features=self.pdm_d_model,
                                        out_features=self.predict_energy_len)

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    self.pdm_seq_len // (self.down_sampling_window ** i),
                    self.pdm_d_model,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
            )       
        self.projection_layer = nn.Linear(self.pdm_d_model, 1,bias=True)
        
        self.last_layer = nn.Softplus() ##fix softplus
        self.use_rev_in = use_rev_in
        
        if use_rev_in:
            self.rev_in_es = RevIN(earthquake_dim)
            self.rev_in_gnss = RevIN(gnss_dim)
            

    def forward(self, earthquake, gnss_data, es_loc=None, gnss_loc=None, es_lap_mx=None, gnss_lap_mx = None, es_geo_mask=None, es_sem_mask=None, gnss_padding_mask=None,gnss_geo_mask=None):

        if self.use_rev_in:
            earthquake = self.rev_in_es(earthquake, mode='norm')
        B, T, N, C = earthquake.size()
        # Earthquake data embedding
        earthquake_list = self.multi_resolution_time_downsampling(earthquake)

        earthquake_enc_list = []
        for i in range(len(earthquake_list)):
            earthquake_enc = self.earthquake_embedding(earthquake_list[i], es_lap_mx, es_loc)
            earthquake_enc_list.append(earthquake_enc)
        # Apply PDM blocks to the multi-resolution earthquake encodings
        for i in range(len(self.pdm_blocks)):
            earthquake_enc_list = self.pdm_blocks[i](earthquake_enc_list)
            earthquake_enc_list = self.node_self_attention_block[i](earthquake_enc_list, es_geo_mask, es_sem_mask, padding_mask = None)
        # earthquake_enc = self.multi_scale_fusion_module(earthquake_enc_list)

        earthquake_enc = self.future_multi_mixing(B, earthquake_enc_list, earthquake_list)

        earthquake_enc = torch.stack(earthquake_enc, dim=-1).sum(-1)

        # GNSS data embedding
        gnss_trend = gnss_data[:,:,:,:self.gnss_dim]
        gnss_season = gnss_data[:,:,:,self.gnss_dim:]
        if self.use_rev_in:
            gnss_trend = self.rev_in_gnss(gnss_trend, mode='norm')
            gnss_season = self.rev_in_gnss(gnss_season, mode='norm')
        gnss_trend = self.gnss_embedding_trend(gnss_trend, gnss_lap_mx, gnss_loc)
        gnss_season = self.gnss_embedding_season(gnss_season, gnss_lap_mx, gnss_loc)

        gnss_trend_enc = 0
        for i, encoder_block in enumerate(self.encoder_blocks_trend):
            gnss_trend = encoder_block(gnss_trend, gnss_geo_mask, padding_mask = gnss_padding_mask)
            gnss_trend_enc += self.skip_convs_trend[i](gnss_trend.permute(0, 3, 2, 1))
        gnss_trend_enc = gnss_trend_enc.permute(0, 3, 2, 1)

        gnss_season_enc = 0
        for i, encoder_block in enumerate(self.encoder_blocks_season):
            gnss_season = encoder_block(gnss_season, gnss_geo_mask, padding_mask = gnss_padding_mask)
            gnss_season_enc += self.skip_convs_season[i](gnss_season.permute(0, 3, 2, 1))
        gnss_season_enc = gnss_season_enc.permute(0, 3, 2, 1)

        gnss_enc = self.gnss_compression(self.layernorm(gnss_season_enc + gnss_trend_enc))
        
        #Cross-attention between earthquake and GNSS data
        ENC = self.cross_attention_network(earthquake_enc, gnss_enc)
        energy = self.predict_energy_head(ENC)
        
        if self.prediction_day_head != 0:
            day = self.predict_day_head(ENC)
        else:
            day = None
        energy = self.last_layer(energy) #B, N, 1

        if self.use_rev_in:
            energy = energy.permute(0, 2, 1)
            energy = energy.unsqueeze(-1)
            energy = self.rev_in_es(energy, mode='denorm')
            energy = energy.squeeze(-1)
            energy = energy.permute(0, 2, 1)
        return energy, day
    
    def future_multi_mixing(self, B, enc_out_list, x_list):
        
        dec_out_list = []
        
        for i, enc_out in zip(range(len(x_list)), enc_out_list):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            dec_out = self.projection_layer(dec_out)
            dec_out = dec_out.squeeze(-1).permute(0, 2, 1)
            dec_out_list.append(dec_out)

        return dec_out_list


        
    
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
    
    
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

    model = ES_net_mixer(earthquake_dim=feature_dim, gnss_dim=feature_dim, embed_dim=embed_dim, lape_dim=lape_dim,earthquake_history_window=1400,
                        down_sampling_method='avg', down_sampling_window=2, down_sampling_layers=3, 
                    pdm_layers=2, pdm_d_model=64, pdm_d_ff=128, 
                    pdm_dropout=0.1, pdm_decomp_method='moving_avg', pdm_moving_avg_kernel=3, 
                    geo_num_heads=4, sem_num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.,
                    mlp_ratio=4., enc_depth=2,type_ln="pre",prediction_day_head=15,prediction_energy_len=240,use_rev_in=True)
    energy, day = model(earthquake=x, gnss_data=gnss_data, es_loc=es_loc, gnss_loc=gnss_loc, es_lap_mx=es_lap_mx, gnss_lap_mx=gnss_lap_mx, es_geo_mask=geo_mask, es_sem_mask=sem_mask, gnss_padding_mask=None,gnss_geo_mask=gnss_geo_mask)
    print(energy.shape, day.shape)
