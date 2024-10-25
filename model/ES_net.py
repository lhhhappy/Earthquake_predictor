import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init
from functools import partial

# Equivalent to dropout
def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # floor
    output = x.div(keep_prob) * random_tensor
    return output

class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.token_embed(x)
        x = self.norm(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        pe = torch.zeros(self.max_len, embed_dim).float()
        pe.requires_grad = False

        position = torch.arange(0, self.max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, T, N, embed_dim)
        # Adjust positional encoding to match the input sequence length
        T = x.size(1)
        N = x.size(2)
        pe = self.pe[:, :T]  # (1, T, embed_dim)
        pe = pe.unsqueeze(2).expand(-1, -1, N, -1)  # (1, T, N, embed_dim)
        return pe.detach()

class LaplacianPE(nn.Module):
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        # lap_mx: (batch_size, N, lape_dim)
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx)  # (batch_size, N, embed_dim)
        lap_pos_enc = lap_pos_enc.unsqueeze(1)  # (batch_size, 1, N, embed_dim)
        return lap_pos_enc

class DataEmbedding(nn.Module):
    def __init__(
        self, feature_dim, embed_dim, lape_dim, drop=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)

        self.position_encoding = PositionalEncoding(embed_dim)
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self, x, lap_mx):
        origin_x = x
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        x += self.position_encoding(x)
        x += self.spatial_embedding(lap_mx)
        x = self.dropout(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

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

class TemporalSelfAttention(nn.Module):
    def __init__(
        self, dim, dim_out, t_attn_size, t_num_heads=6, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'),
    ):
        super().__init__()
        assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.device = device
        self.t_attn_size = t_attn_size

        self.t_q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, N, D = x.shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale

        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)

        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)

        x = self.proj(t_x)
        x = self.proj_drop(x)
        return x

class STSelfAttention(nn.Module):
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, qkv_bias=False,
        attn_drop=0., proj_drop=0., output_dim=1):
        super().__init__()
        assert dim % (geo_num_heads + sem_num_heads + t_num_heads) == 0
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.t_num_heads = t_num_heads
        self.head_dim = dim // (geo_num_heads + sem_num_heads + t_num_heads)
        self.scale = self.head_dim ** -0.5
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        self.geo_ratio = geo_num_heads / (geo_num_heads + sem_num_heads + t_num_heads) if geo_num_heads > 0 else 0
        self.sem_ratio = sem_num_heads / (geo_num_heads + sem_num_heads + t_num_heads) if sem_num_heads > 0 else 0
        self.t_ratio = 1 - self.geo_ratio - self.sem_ratio
        self.output_dim = output_dim

        # Conditional initialization based on the number of heads
        self.geo_q_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias) if geo_num_heads > 0 else None
        self.geo_k_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias) if geo_num_heads > 0 else None
        self.geo_v_conv = nn.Conv2d(dim, int(dim * self.geo_ratio), kernel_size=1, bias=qkv_bias) if geo_num_heads > 0 else None
        self.geo_attn_drop = nn.Dropout(attn_drop) if geo_num_heads > 0 else None

        self.sem_q_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias) if sem_num_heads > 0 else None
        self.sem_k_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias) if sem_num_heads > 0 else None
        self.sem_v_conv = nn.Conv2d(dim, int(dim * self.sem_ratio), kernel_size=1, bias=qkv_bias) if sem_num_heads > 0 else None
        self.sem_attn_drop = nn.Dropout(attn_drop) if sem_num_heads > 0 else None

        self.t_q_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, int(dim * self.t_ratio), kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, geo_mask=None, sem_mask=None):
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
            geo_attn = self.geo_attn_drop(geo_attn)
            geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, int(D * self.geo_ratio))
        else:
            geo_x = torch.zeros(B, T, N, int(D * self.geo_ratio), device=x.device)

        # Semantic block
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
            sem_attn = self.sem_attn_drop(sem_attn)
            sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, int(D * self.sem_ratio))
        else:
            sem_x = torch.zeros(B, T, N, int(D * self.sem_ratio), device=x.device)

        # Concatenate and project output
        x = self.proj(torch.cat([t_x, geo_x, sem_x], dim=-1))
        x = self.proj_drop(x)
        return x

class STEncoderBlock(nn.Module):

    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, t_num_heads=2, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, type_ln="pre", output_dim=1,
    ):
        super().__init__()
        self.type_ln = type_ln
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(
            dim, s_attn_size, t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, output_dim=output_dim,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, geo_mask=None, sem_mask=None):
        if self.type_ln == 'pre':
            x = x + self.drop_path(self.st_attn(self.norm1(x), geo_mask=geo_mask, sem_mask=sem_mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.type_ln == 'post':
            x = self.norm1(x + self.drop_path(self.st_attn(x, geo_mask=geo_mask, sem_mask=sem_mask)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x

class PatchMapping(nn.Module):
    def __init__(self, patch_len):
        super(PatchMapping, self).__init__()
        self.patch_len = patch_len
        # Use a linear layer to extract features from each patch.
        # The input dimension is the feature dimension of the patch, and the output dimension is 1.
        self.linear = nn.Linear(patch_len, 1)

    def forward(self, x):
        batch_size, T, N, domdel = x.shape
        # Reshape the input to handle patches.
        x = x.view(batch_size, T // self.patch_len, self.patch_len, N, domdel)
        # Pass the features along the patch_len dimension to the linear layer for transformation.
        x = self.linear(x.permute(0, 1, 3, 4, 2))  # Adjust dimensions to match the input of the linear layer: (batch_size, T // patch_len, N, domdel, patch_len)
        x = x.permute(0, 1, 4, 2, 3)  # Adjust dimensions to match subsequent processing: (batch_size, T // patch_len, embed_dim, N, domdel)
        x = x.view(batch_size, T // self.patch_len, N, domdel)
        return x

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super(CrossAttention, self).__init__()
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)
        self.scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    def forward(self, A, B):
        B_size, T, N1, d = A.shape
        _, _, N2, _ = B.shape

        # 1. Linear mapping
        Q = self.W_q(A)  # (B, T, N1, d_k)
        K = self.W_k(B)  # (B, T, N2, d_k)
        V = self.W_v(B)  # (B, T, N2, d_v)

        # 2. Calculate attention scores
        # Reshape to match matrix multiplication requirements
        Q = Q.view(B_size * T, N1, -1)  # (B*T, N1, d_k)
        K = K.view(B_size * T, N2, -1)  # (B*T, N2, d_k)
        V = V.view(B_size * T, N2, -1)  # (B*T, N2, d_v)

        # Compute the dot product of Q and the transpose of K
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B*T, N1, N2)

        # 3. Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B*T, N1, N2)

        # 4. Weighted sum of values
        output = torch.bmm(attn_weights, V)  # (B*T, N1, d_v)

        # Restore original shape
        output = output.view(B_size, T, N1, -1)  # (B, T, N1, d_v)
        return output

class TemporalConvDecoder(nn.Module):
    def __init__(self, d_model, num_layers, kernel_size=3, dropout=0.1, output_dim=1):
        """
        Decoder layer using 1D convolutions.

        :param d_model: Feature dimension.
        :param num_layers: Number of convolutional layers.
        :param kernel_size: Size of the convolution kernel.
        :param dropout: Dropout probability.
        :param output_dim: Output dimension of the final linear layer.
        """
        super(TemporalConvDecoder, self).__init__()

        # Define multiple 1D convolutional layers.
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=kernel_size//2)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

        # Final fully connected layer to map from the convolution result to the output dimension.
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, output_dim)
        self.energy_fc = nn.Linear(1, d_model)
    def forward(self, x, energy):
        """
        Forward pass.

        :param x: Input tensor of shape (batch, T, N, d_model)
        :param energy: Energy tensor of shape (batch, N, 1)
        :return: Output tensor of shape (batch, N, output_dim)
        """
        batch_size, T, N, d_model = x.size()

        # Reshape input to match Conv1d's expected input: (batch * N, d_model, T)
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch, N, d_model, T)
        x = x.view(batch_size * N, d_model, T)  # (batch * N, d_model, T)

        # Apply multiple 1D convolutional layers to integrate temporal information.
        for conv in self.temporal_convs:
            x = self.activation(conv(x))  # (batch * N, d_model, T)
            x = self.dropout(x)

        # Aggregate features over the time dimension using max pooling.
        x, _ = x.max(dim=2)  # (batch * N, d_model)

        # Reshape back to (batch, N, d_model).
        x = x.view(batch_size, N, d_model)

        # Apply the fully connected layer to further integrate information.
        x = self.fc1(x) + self.energy_fc(energy)
        x = self.activation(x)
        x = self.fc2(x)  # (batch, N, output_dim)

        return x

class ES_net(nn.Module):
    def __init__(self, feature_dim, ext_dim, gnss_feature_dim,
                 embed_dim=64, skip_dim=256, lape_dim=8, geo_num_heads=4,
                 sem_num_heads=2, t_num_heads=2, mlp_ratio=4, qkv_bias=True, drop=0.,
                 attn_drop=0., drop_path=0.3, s_attn_size=3, t_attn_size=3, enc_depth=6,
                 type_ln="pre", output_dim=1, input_window=12,
                 output_window=1, predict_day_class=14,
                 far_mask_delta=0, dtw_delta=0):

        super().__init__()

        self.feature_dim = feature_dim
        self.ext_dim = ext_dim

        self.embed_dim = embed_dim
        self.skip_dim = skip_dim
        self.lape_dim = lape_dim
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.t_num_heads = t_num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        self.enc_depth = enc_depth
        self.type_ln = type_ln

        self.output_dim = output_dim
        self.input_window = input_window
        self.output_window = output_window

        self.far_mask_delta = far_mask_delta
        self.dtw_delta = dtw_delta
        self.gnss_feature_dim = gnss_feature_dim
        self.predict_day_class = predict_day_class

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, drop=self.drop,)

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]

        self.encoder_blocks_energy = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size,
                geo_num_heads=self.geo_num_heads, sem_num_heads=self.sem_num_heads, t_num_heads=self.t_num_heads,
                mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop, attn_drop=self.attn_drop,
                drop_path=enc_dpr[i], act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), type_ln=self.type_ln, output_dim=self.output_dim,
            ) for i in range(enc_depth)
        ])

        self.gnss_embed_layer = DataEmbedding(
            self.gnss_feature_dim, self.embed_dim, lape_dim, drop=self.drop,)

        self.encoder_blocks_gnss = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size,
                geo_num_heads=self.geo_num_heads + self.sem_num_heads, sem_num_heads=0, t_num_heads=self.t_num_heads,
                mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, drop=self.drop, attn_drop=self.attn_drop,
                drop_path=enc_dpr[i], act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6), type_ln=self.type_ln, output_dim=self.output_dim,
            ) for i in range(enc_depth)
        ])

        self.skip_convs_energy = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

        self.skip_convs_gnss = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

        self.es_conv1 = nn.Conv2d(
            in_channels=self.input_window, out_channels=self.output_window, kernel_size=1, bias=True,
        )
        self.es_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True,
        )

        self.day2week = PatchMapping(self.predict_day_class)

        self.cross_attn = CrossAttention(self.skip_dim, self.skip_dim, self.skip_dim)

        self.day_deocder = TemporalConvDecoder(self.skip_dim, num_layers=2, output_dim=self.predict_day_class)

    def forward(self, x, gnss, lap_mx=None, gnss_lap_mx=None, es_geo_mask=None, es_sem_mask=None, gnss_geo_mask=None):
        T = x.shape[1]

        enc = self.enc_embed_layer(x, lap_mx)

        gnss_enc = self.gnss_embed_layer(gnss, gnss_lap_mx)

        skip_earthquake = 0
        for i, encoder_block in enumerate(self.encoder_blocks_energy):
            enc = encoder_block(enc, es_geo_mask, es_sem_mask)
            skip_earthquake += self.skip_convs_energy[i](enc.permute(0, 3, 2, 1))

        skip_gnss = 0
        for i, encoder_block in enumerate(self.encoder_blocks_gnss):
            gnss_enc = encoder_block(gnss_enc, gnss_geo_mask)
            skip_gnss += self.skip_convs_gnss[i](gnss_enc.permute(0, 3, 2, 1))

        skip_earthquake = skip_earthquake.permute(0, 3, 2, 1)
        skip_gnss_day = skip_gnss.permute(0, 3, 2, 1)

        skip_gnss_week = self.day2week(skip_gnss_day)

        ENC_ST = self.cross_attn(skip_earthquake, skip_gnss_week) + skip_earthquake
        print(ENC_ST.shape)
        energy_predict = self.es_conv1(F.relu(ENC_ST))
        energy_predict = self.es_conv2(F.relu(energy_predict.permute(0, 3, 2, 1))).permute(0, 3, 2, 1).squeeze(1)

        predict_day = self.day_deocder(ENC_ST, energy_predict.detach())
        return energy_predict, predict_day