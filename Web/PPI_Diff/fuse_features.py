import torch
import torch.nn as nn
import numpy as np
import os
import gc
import esm
from tqdm import tqdm

# ==============================================================
# --- 全局设置与 ESM 模型管理 ---
# ==============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 全局变量占位
esm_model = None
esm_alphabet = None
esm_batch_converter = None
esm_embedding_dim = 0
ESM_LOADED = False

def init_esm_model(model_name="esm2_t6_8M_UR50D"):
    """
    显式初始化 ESM 模型。
    """
    global esm_model, esm_alphabet, esm_batch_converter, esm_embedding_dim, ESM_LOADED
    
    if ESM_LOADED:
        return

    print(f"--- 正在初始化 ESM 模型: {model_name} ---")
    try:
        esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        esm_model.eval()
        esm_model = esm_model.to(device)
        esm_batch_converter = esm_alphabet.get_batch_converter()
        esm_embedding_dim = esm_model.embed_dim
        ESM_LOADED = True
        print(f"ESM-2 模型加载完成。嵌入维度: {esm_embedding_dim}")
    except Exception as e:
        print(f"警告：加载 ESM-2 模型时出错: {e}。将跳过 ESM 嵌入。")
        ESM_LOADED = False

# ==============================================================
# --- 核心融合函数 (fuse_target_features) ---
# ==============================================================
def fuse_target_features(npz_files, target_uniprot_id, use_esm=True, use_transformer=False, ablate_resolution=False):
    """
    Args:
        ablate_resolution (bool): 如果为 True，强制忽略分辨率信息（设为 NaN），这将导致退化为简单的平均融合。
    """
    
    # --- 1. 加载数据 ---
    all_data = []
    min_pos = float('inf')
    max_pos = float('-inf')
    feature_dim = -1

    for npz_file in npz_files:
        try:
            data = np.load(npz_file, allow_pickle=True)
            fused_features = torch.tensor(data['fused_features'], dtype=torch.float32, device=device)
            coords = torch.tensor(data['coords'], dtype=torch.float32, device=device)
            sequence = str(data['sequence'])
            absolute_positions = torch.tensor(data['absolute_positions'], dtype=torch.float32, device=device).reshape(-1, 1)
            
            # --- 处理分辨率 (消融逻辑) ---
            raw_resolution = float(data.get('resolution', np.nan))
            
            if ablate_resolution:
                resolution = np.nan
            else:
                # 过滤异常分辨率，通常 PDB 分辨率在 0.5 - 10.0 之间
                if np.isnan(raw_resolution) or not (0.0 < raw_resolution < 100.0):
                    resolution = np.nan
                else:
                    resolution = raw_resolution

            if feature_dim == -1:
                feature_dim = fused_features.shape[1]
            elif fused_features.shape[1] != feature_dim:
                continue

            if absolute_positions.size(0) > 0:
                current_min_pos = int(torch.min(absolute_positions).item())
                current_max_pos = int(torch.max(absolute_positions).item())
                min_pos = min(min_pos, current_min_pos)
                max_pos = max(max_pos, current_max_pos)

            all_data.append({
                'fused_features': fused_features,
                'coords': coords,
                'sequence': sequence,
                'absolute_positions': absolute_positions.squeeze().tolist(),
                'resolution': resolution
            })
        except Exception:
            continue

    if not all_data:
        return None

    seq_len = max_pos - min_pos + 1
    if seq_len <= 0: return None

    # --- 2. 创建 Placeholders ---
    # Shape: [SeqLen, Num_Files, FeatureDim]
    features_per_pos = torch.full((seq_len, len(all_data), feature_dim), float('nan'), dtype=torch.float32, device=device)
    # Shape: [SeqLen, Num_Files]
    resolutions_per_pos = torch.full((seq_len, len(all_data)), float('nan'), dtype=torch.float32, device=device)
    
    best_res_coords = torch.full((seq_len, 3), float('nan'), dtype=torch.float32, device=device)
    final_sequence_list = ['-'] * seq_len
    
    # 记录每个位置是否有有效数据
    mask_per_pos = torch.zeros((seq_len, len(all_data)), dtype=torch.bool, device=device)

    # --- 3. 填充 Placeholders ---
    for struct_idx, data in enumerate(all_data):
        absolute_positions = data['absolute_positions']
        fused_features = data['fused_features']
        coords = data['coords']
        sequence = data['sequence']
        resolution = data['resolution']

        if isinstance(absolute_positions, float): absolute_positions = [absolute_positions]
        if len(absolute_positions) != fused_features.shape[0]: continue

        for res_idx_in_struct, pdb_res_num_float in enumerate(absolute_positions):
            aligned_pos = int(pdb_res_num_float) - min_pos
            if 0 <= aligned_pos < seq_len:
                features_per_pos[aligned_pos, struct_idx, :] = fused_features[res_idx_in_struct, :]
                mask_per_pos[aligned_pos, struct_idx] = True
                resolutions_per_pos[aligned_pos, struct_idx] = resolution
                
                # 填充序列
                if final_sequence_list[aligned_pos] == '-':
                    final_sequence_list[aligned_pos] = sequence[res_idx_in_struct]
                
                # 坐标填充：这里可以优化为取“分辨率最高”的那个坐标，而不是第一个
                # 为了保持简单，先沿用逻辑，但在 return 中我们会返回 best_resolutions
                if torch.isnan(best_res_coords[aligned_pos]).any():
                    best_res_coords[aligned_pos, :] = coords[res_idx_in_struct, :]

    # ==============================================================================
    # --- 4. [修改重点] 分辨率加权特征融合 (Resolution-Weighted Feature Fusion) ---
    # ==============================================================================
    
    fused_struct_features = torch.full((seq_len, feature_dim), float('nan'), dtype=torch.float32, device=device)
    fusion_run_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)

    # A. 计算权重 (Weight Calculation)
    # 逻辑：分辨率越小(Angstrom)，质量越高，权重应越大。
    # 使用反比权重: weight = 1 / (resolution + epsilon)
    # 如果分辨率缺失 (NaN)，给予一个极小的默认权重，保证不报错但也不占主导。
    
    # 填充 NaN 为一个较大的值 (例如 50.0埃，表示质量很差)，防止计算权重时出错
    # 注意：我们稍后会用 mask_per_pos 过滤掉根本不存在数据的地方
    res_filled = torch.nan_to_num(resolutions_per_pos, nan=50.0)
    
    # 计算基础权重
    weights = 1.0 / (res_filled + 1e-6) # Shape: [SeqLen, Num_Files]
    
    # 应用 Mask：如果该文件在这个位置没有数据，权重强制为 0
    weights = weights * mask_per_pos.float()
    
    # B. 加权求和 (Weighted Sum)
    # 扩展权重维度以匹配特征: [SeqLen, Num_Files, 1]
    weights_expanded = weights.unsqueeze(-1)
    
    # 把特征中的 NaN 变成 0，方便乘法
    features_safe = torch.nan_to_num(features_per_pos, nan=0.0)
    
    # 分子：Sum(Feature * Weight)
    weighted_sum_features = torch.sum(features_safe * weights_expanded, dim=1) # [SeqLen, FeatureDim]
    
    # 分母：Sum(Weight)
    sum_weights = torch.sum(weights_expanded, dim=1) # [SeqLen, 1]
    
    # 避免分母为 0 (即该位置没有任何有效来源)
    valid_pos_mask = (sum_weights.squeeze(-1) > 1e-9)
    
    # C. 计算最终特征
    fused_struct_features[valid_pos_mask] = weighted_sum_features[valid_pos_mask] / sum_weights[valid_pos_mask]
    fusion_run_mask[valid_pos_mask] = True

    # ==============================================================================
    # --- 5. [修改重点] 提取代表性分辨率 (Extract Representative Resolution) ---
    # ==============================================================================
    # 我们需要告诉模型，这个位置的数据到底有多“保真”。
    # 通常取所有来源中最好的（最小的）分辨率作为该位置的代表。
    
    # 将 NaN 替换为无穷大，以便求 min
    res_for_min = torch.nan_to_num(resolutions_per_pos, nan=float('inf'))
    
    # 取最小分辨率 (Best Resolution)
    best_resolutions_per_residue, _ = torch.min(res_for_min, dim=1) # [SeqLen]
    
    # 恢复无效值 (如果全是 inf，说明没数据，变回 NaN 或默认值)
    # 这里我们保留 inf 或者设为 0，取决于后续模型怎么处理。
    # 建议：在后续步骤标准化，这里先保留原始值。但为了避免后续 tensor 处理麻烦，
    # 我们把 inf 设为 0 (表示无数据/Masked) 或者一个标记值。
    # 这里我们只保留 valid_indices 的部分，所以不用太担心 inf。

    # 清理内存
    del features_per_pos, mask_per_pos, resolutions_per_pos, features_safe, weights, weights_expanded
    gc.collect()

    # --- 6. 过滤无效位置 ---
    coords_exist_mask = ~torch.isnan(best_res_coords).any(dim=1)
    sequence_exist_mask = torch.tensor([aa != '-' for aa in final_sequence_list], dtype=torch.bool, device=device)
    
    final_valid_mask = fusion_run_mask & coords_exist_mask & sequence_exist_mask
    valid_indices = final_valid_mask.nonzero(as_tuple=True)[0]

    if len(valid_indices) == 0: return None

    final_struct_features = fused_struct_features[valid_indices]
    final_coords = best_res_coords[valid_indices]
    final_resolutions = best_resolutions_per_residue[valid_indices] # <--- 提取对应的分辨率
    final_valid_sequence = "".join([final_sequence_list[k.item()] for k in valid_indices])
    final_original_indices = (min_pos + valid_indices.cpu().numpy()).tolist()

    del fused_struct_features, best_res_coords, best_resolutions_per_residue
    gc.collect()

    # --- 7. ESM Embeddings ---
    final_esm_embeddings = None
    if use_esm and ESM_LOADED:
        if final_valid_sequence:
            try:
                esm_input_data = [("protein_segment", final_valid_sequence)]
                _, _, esm_batch_tokens = esm_batch_converter(esm_input_data)
                esm_batch_tokens = esm_batch_tokens.to(device)

                with torch.no_grad():
                    results = esm_model(esm_batch_tokens, repr_layers=[esm_model.num_layers], return_contacts=False)
                
                token_representations = results["representations"][esm_model.num_layers]
                if token_representations.shape[1] == len(final_valid_sequence) + 2:
                    final_esm_embeddings = token_representations.squeeze(0)[1:-1, :]
                else:
                    final_esm_embeddings = None
                
                del results, token_representations, esm_batch_tokens
                gc.collect()
            except Exception as e:
                print(f"ESM Extract Error: {e}")
                final_esm_embeddings = None

    # --- 8. 结合特征 ---
    if final_esm_embeddings is not None and final_struct_features.shape[0] == final_esm_embeddings.shape[0]:
        combined_features = torch.cat([final_struct_features, final_esm_embeddings], dim=-1)
    else:
        combined_features = final_struct_features

    if final_esm_embeddings is not None:
        del final_struct_features
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "final_features": combined_features,
        "coords": final_coords,
        "sequence": final_valid_sequence,
        "original_indices": final_original_indices,
        "resolution_score": final_resolutions  # <--- [新增] 返回分辨率分数
    }