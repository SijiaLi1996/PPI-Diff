import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from tqdm import tqdm

# --- 辅助：预计算 DDPM 参数 (放在全局或类里均可) ---
# 为了方便，我们在 sample 函数内部计算，保证独立性
def get_ddpm_schedule(timesteps, device):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    return betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class ConditionalAngleDiffusion(nn.Module):
    def __init__(self, angle_dim, context_dim, hidden_dim, timesteps, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.angle_dim = angle_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps
        
        self.time_embedding = nn.Embedding(timesteps, hidden_dim)
        self.angle_projection = nn.Linear(angle_dim, hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # 修正：分辨率嵌入增加 LayerNorm 以防止数值爆炸
        self.resolution_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim) # 添加归一化
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, angle_dim)
        
        # 修正：移除 gain=0.1，使用默认初始化或更合理的初始化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 修正：使用 gain=1.0 (Xavier默认) 或 Kaiming
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0) 
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x_angles, context, t, resolution, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # ... (前面的代码保持不变，省略以节省空间，直接到 try 块) ...
        batch_size = x_angles.shape[0]
        tgt_seq_len = x_angles.shape[1]
        mem_seq_len = context.shape[1]

        # 简单的掩码处理逻辑 (保留您的代码逻辑)
        # ... (建议确保 train.py 传入的 mask 维度正确，这里不再赘述) ...

        try:
            x = self.angle_projection(x_angles)
            memory = self.context_projection(context)
            
            if resolution.dtype != torch.float32: resolution = resolution.float()
            # ... (分辨率维度调整逻辑保持不变) ...
            if resolution.dim() == 2: resolution = resolution.unsqueeze(-1)
            # 此时 resolution: [B, L, 1]

            res_emb = self.resolution_embedding(resolution)
            
            # 关键点：将分辨率加到 Memory 上
            memory = memory + res_emb 

            x = self.pos_encoder(x)
            memory = self.pos_encoder(memory)
            
            time_emb = self.time_embedding(t).unsqueeze(1)
            x = x + time_emb
            
            # Transformer Decoder: 
            # tgt=x (Angles), memory=memory (Context + Resolution)
            # Cross Attention 发生在这里
            output = self.transformer_decoder(
                tgt=x, 
                memory=memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            return self.output_projection(output)
            
        except Exception as e:
            logging.error(f"Diffusion forward error: {e}")
            raise e

class SequencePredictor(nn.Module):
    def __init__(self, angle_dim, context_dim, hidden_dim, vocab_size, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        # ... (初始化代码同上) ...
        self.angle_dim = angle_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        self.angle_projection = nn.Linear(angle_dim, hidden_dim)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        self.resolution_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # 修正：初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 修正：gain=1.0
            torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

    def forward(self, x_angles, context, resolution, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 逻辑与 Diffusion 几乎一致，只是没有 Time Embedding
        try:
            x = self.angle_projection(x_angles)
            memory = self.context_projection(context)
            
            if resolution.dtype != torch.float32: resolution = resolution.float()
            if resolution.dim() == 2: resolution = resolution.unsqueeze(-1)
            
            res_emb = self.resolution_embedding(resolution)
            memory = memory + res_emb
            
            x = self.pos_encoder(x)
            memory = self.pos_encoder(memory)
            
            # Cross Attention: Sequence Decoder 关注 Context
            output = self.transformer_decoder(
                tgt=x, 
                memory=memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            return self.output_projection(output)
        except Exception as e:
            raise e

# ==========================================================
# 修复后的采样函数 (DDPM 逻辑)
# ==========================================================

@torch.no_grad()
def sample(model, context, resolution, sequence_length, batch_size=1, timesteps=1000, context_mask=None):
    """
    标准的 DDPM 采样循环
    """
    device = next(model.parameters()).device
    model.eval()
    
    # 1. 准备参数
    if isinstance(context, torch.Tensor): context = context.to(device)
    if isinstance(resolution, torch.Tensor): resolution = resolution.to(device)
    if context_mask is not None: context_mask = context_mask.to(device)
    
    # 获取 DDPM 参数 (Alpha, Beta)
    betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas = get_ddpm_schedule(timesteps, device)
    
    # 2. 从高斯噪声开始 x_T
    angle_dim = model.angle_projection.in_features
    shape = (batch_size, sequence_length, angle_dim)
    img = torch.randn(shape, device=device) # x_t
    
    # 3. 逐步去噪
    for i in tqdm(reversed(range(0, timesteps)), desc="Sampling", total=timesteps, leave=False):
        t_tensor = torch.full((batch_size,), i, device=device, dtype=torch.long)
        
        # A. 预测噪声 epsilon_theta
        predicted_noise = model(
            x_angles=img, 
            context=context, 
            t=t_tensor, 
            resolution=resolution,
            memory_key_padding_mask=context_mask
        )
        
        # B. 计算系数
        beta_t = betas[i]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[i]
        sqrt_recip_alpha_t = sqrt_recip_alphas[i]
        
        # C. DDPM 更新公式: x_{t-1} = (1/sqrt(alpha)) * (x_t - coeff * eps)
        # posterior_mean
        model_mean = sqrt_recip_alpha_t * (
            img - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
        )
        
        # D. 添加方差 (除了最后一步 t=0)
        if i > 0:
            posterior_variance = beta_t # 简化版方差
            noise = torch.randn_like(img)
            img = model_mean + torch.sqrt(posterior_variance) * noise
        else:
            img = model_mean
            
    return img

# DDIM 采样也需要重写，暂时建议先用 sample 跑通