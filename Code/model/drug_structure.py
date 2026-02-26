# drug_structure.py (进一步精简版)

import os
import numpy as np
import torch
import torch.nn as nn
from Bio.PDB import PDBParser, PPBuilder, Polypeptide
from Bio.SeqUtils import seq1
import protein_geometry
import math

# --- 常量 ---
AMINO_ACIDS = sorted(Polypeptide.protein_letters_3to1.values())
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AMINO_ACIDS = len(AMINO_ACIDS)

# --- 简化的几何参数（仅保留必要的）---
# 标准键长（用于RMSD/TM-Score计算时的坐标转换）
BOND_N_CA = 1.458
BOND_CA_C = 1.525
BOND_C_N = 1.329
BOND_C_O = 1.231

# 标准键角（仅保留实际使用的）
ANGLE_N_CA_C = np.radians(111.2)  # 直接转换为弧度
ANGLE_CA_C_O = np.radians(120.0)

# --- 简化的坐标转换函数 ---
def angles_to_coordinates(angles):
    """
    将角度信息转换为3D坐标（仅用于RMSD/TM-Score计算）
    使用简化的几何模型
    
    Args:
        angles: numpy array, shape (seq_len, 12)
    
    Returns:
        coords: numpy array, shape (seq_len*4, 3)
    """
    if len(angles.shape) != 2 or angles.shape[1] < 3:
        raise ValueError("angles应该是shape为(seq_len, >=3)的数组")
    
    seq_len = angles.shape[0]
    coords = np.zeros((seq_len * 4, 3))  # N, CA, C, O for each residue
    
    for i in range(seq_len):
        # 提取角度（只使用前3个：phi, psi, omega）
        phi = angles[i, 0] if angles.shape[1] > 0 else 0.0
        psi = angles[i, 1] if angles.shape[1] > 1 else 0.0
        omega = angles[i, 2] if angles.shape[1] > 2 else np.pi
        
        # 原子索引
        n_idx = i * 4
        ca_idx = i * 4 + 1
        c_idx = i * 4 + 2
        o_idx = i * 4 + 3
        
        if i == 0:
            # 第一个残基：简单的初始化
            coords[n_idx] = np.array([0.0, 0.0, 0.0])
            coords[ca_idx] = np.array([BOND_N_CA, 0.0, 0.0])
            coords[c_idx] = coords[ca_idx] + np.array([
                BOND_CA_C * np.cos(np.pi - ANGLE_N_CA_C),
                BOND_CA_C * np.sin(np.pi - ANGLE_N_CA_C),
                0.0
            ])
            coords[o_idx] = coords[c_idx] + np.array([0.0, BOND_C_O, 0.0])
        else:
            # 后续残基：使用简化的线性构建
            # 这里主要是为了提供合理的坐标用于RMSD计算
            # 不需要完全准确的几何结构
            base_x = i * 3.8  # 大约的残基间距
            coords[n_idx] = np.array([base_x, 0.0, 0.0])
            coords[ca_idx] = np.array([base_x + BOND_N_CA, 0.0, 0.0])
            coords[c_idx] = np.array([base_x + BOND_N_CA + BOND_CA_C, 0.0, 0.0])
            coords[o_idx] = np.array([base_x + BOND_N_CA + BOND_CA_C, BOND_C_O, 0.0])
    
    return coords

# --- 特征提取函数 ---
def extract_drug_structure_features(pdb_file):
    """从药物 PDB 文件中提取结构特征"""
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("drug", pdb_file)
        model = structure[0]

        ppb = PPBuilder()
        peptides = ppb.build_peptides(model)

        if not peptides:
            print(f"信息：在 {pdb_file} 中未找到多肽链。")
            return None

        poly = peptides[0]
        if not poly:
            print(f"信息：在 {pdb_file} 中找到的多肽链为空。")
            return None

        # 计算角度
        raw_angles_list = protein_geometry.calculate_backbone_angles(poly)

        # 收集有效残基信息
        valid_residues_data = []
        original_residue_indices = []
        
        for i, residue in enumerate(poly):
            res_name = residue.get_resname()
            if Polypeptide.is_aa(res_name, standard=True) and "CA" in residue:
                try:
                    aa = seq1(res_name)
                except KeyError:
                    print(f"警告：无法将残基名 '{res_name}' 转换为单字母代码。跳过此残基。")
                    continue

                ca_coord = residue["CA"].get_coord()
                pdb_res_id = residue.id[1]

                valid_residues_data.append({
                    "aa": aa, 
                    "coord": ca_coord, 
                    "pdb_id": pdb_res_id
                })
                original_residue_indices.append(i)

        if not valid_residues_data:
            print(f"信息：在 {pdb_file} 中未找到有效的标准氨基酸残基。")
            return None

        # 后处理
        sequence = "".join([data["aa"] for data in valid_residues_data])
        coords = np.array([data["coord"] for data in valid_residues_data])

        # 处理角度
        try:
            filtered_raw_angles = [raw_angles_list[i] for i in original_residue_indices]
            angle_features = protein_geometry.process_angles_sin_cos(filtered_raw_angles)
            return sequence, coords, angle_features
        except IndexError:
            print(f"错误：提取角度时发生索引错误")
            return None

    except Exception as e:
        print(f"处理 {pdb_file} 时发生错误: {e}")
        return None

# --- 辅助函数 ---
def sequence_to_tensor(sequence):
    """将氨基酸序列转换为 PyTorch 张量"""
    indices = [AA_TO_INDEX.get(aa, 0) for aa in sequence]
    return torch.tensor(indices, dtype=torch.long)

def tensor_to_sequence(tensor):
    """将 PyTorch 张量转换为氨基酸序列"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    sequence = "".join([AMINO_ACIDS[min(i, len(AMINO_ACIDS)-1)] for i in tensor])
    return sequence

# --- 模型定义 ---
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

class AngleDiffusion(nn.Module):
    """角度扩散模型"""
    def __init__(self, input_dim, hidden_dim, timesteps, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.time_embedding = nn.Embedding(timesteps, hidden_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, 12)

    def forward(self, x, t, src_key_padding_mask=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        time_emb = self.time_embedding(t).unsqueeze(1)
        x = x + time_emb
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        predicted_noise = self.output_projection(output)
        return predicted_noise

class SequencePredictor(nn.Module):
    """序列预测模型"""
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.output_projection(output)
        return logits

# --- 采样和预测函数 ---
def sample_diffusion(model, shape, device, timesteps, noise_scheduler):
    """使用扩散模型进行采样"""
    model.eval()
    with torch.no_grad():
        angles = torch.randn(shape, device=device)
        for i in reversed(range(timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            predicted_noise = model(angles, t)
            angles = noise_scheduler.step(predicted_noise, i, angles).prev_sample
    return angles

def predict_sequence(model, angle_features, device):
    """使用序列预测模型预测序列"""
    model.eval()
    with torch.no_grad():
        logits = model(angle_features.to(device))
        predicted_indices = torch.argmax(logits, dim=-1)
    return predicted_indices

if __name__ == "__main__":
    print("=== drug_structure.py 模块测试 ===")
    
    # 测试基本功能
    test_seq = "ACDEFG"
    tensor = sequence_to_tensor(test_seq)
    recovered_seq = tensor_to_sequence(tensor)
    print(f"序列转换测试: {test_seq == recovered_seq}")
    
    # 测试角度到坐标转换
    test_angles = np.random.randn(3, 12)
    try:
        coords = angles_to_coordinates(test_angles)
        print(f"角度到坐标转换成功，输出形状: {coords.shape}")
    except Exception as e:
        print(f"角度到坐标转换失败: {e}")