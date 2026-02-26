import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import glob
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

# 导入您的工具函数
from utils import _find_pdb_file 
from drug_structure import extract_drug_structure_features, sequence_to_tensor
from fuse_features import fuse_target_features, init_esm_model

class ProteinInteractionDataset(Dataset):
    def __init__(self, links_file, features_raw_dir, pdb_dir, cache_root="./cached_data", 
                 score_threshold=0, max_samples=None, sep='\t',
                 use_esm=True, ablate_resolution=False, bidirectional=True):
        """
        修复版 Dataset:
        1. 读取并传递真实的分辨率数据 (resolution_score)
        2. 支持 ablate_resolution 参数：如果为 True，强制将分辨率设为 0
        """
        self.pdb_dir = pdb_dir
        self.ablate_resolution = ablate_resolution # 保存参数
        
        # --- 1. 确定缓存目录名称 ---
        config_parts = []
        config_parts.append("with_esm" if use_esm else "no_esm")
        config_parts.append("no_res" if ablate_resolution else "with_res")
        
        self.cache_dir = os.path.join(cache_root, "_".join(config_parts))
        os.makedirs(self.cache_dir, exist_ok=True)

        # --- 2. 加载关系列表 ---
        print(f"📖 读取链接文件: {links_file}")
        df = pd.read_csv(links_file, sep=sep)
        if 'score' in df.columns:
            df = df[df['score'] > score_threshold]
        
        # --- 3. 自动预处理与验证 ---
        unique_proteins = set(df['protein1_uniprot_id'].unique()) | set(df['protein2_uniprot_id'].unique())
        
        if use_esm:
            init_esm_model() 

        print(f"🔍 正在验证 {len(unique_proteins)} 个蛋白质的特征缓存...")
        
        self.valid_ids = set()

        for prot_id in tqdm(unique_proteins, desc="Verifying & Caching"):
            prot_id = str(prot_id).strip()
            cache_path = os.path.join(self.cache_dir, f"{prot_id}.npz")
            
            # A. 检查缓存是否已存在
            if os.path.exists(cache_path):
                self.valid_ids.add(prot_id)
                continue
            
            # B. 缓存不存在，去原始目录找 (支持模糊搜索)
            search_pattern = os.path.join(features_raw_dir, prot_id, "*.npz")
            feature_files = sorted(glob.glob(search_pattern))
            
            # 兜底: 找 features_raw_dir/Q9ULZ1.npz
            if not feature_files:
                flat_path = os.path.join(features_raw_dir, f"{prot_id}.npz")
                if os.path.exists(flat_path):
                    feature_files = [flat_path]
            
            # C. 如果找到了原始文件 -> 生成缓存
            if feature_files:
                try:
                    result = fuse_target_features(
                        feature_files, 
                        prot_id, 
                        use_esm=use_esm, 
                        use_transformer=False, # 这里的 transformer 已经弃用
                        ablate_resolution=ablate_resolution
                    )
                    
                    if result is not None:
                        save_dict = {}
                        for k, v in result.items():
                            if isinstance(v, torch.Tensor):
                                save_dict[k] = v.detach().cpu().numpy()
                            else:
                                save_dict[k] = v
                        np.savez_compressed(cache_path, **save_dict)
                        
                        self.valid_ids.add(prot_id)
                except Exception as e:
                    print(f"❌ 处理失败 {prot_id}: {e}")
                    pass
            else:
                pass

        print(f"✅ 有效蛋白质文件: {len(self.valid_ids)} / {len(unique_proteins)}")

        # --- 4. 构建样本列表 ---
        self.samples = []
        skipped_count = 0
        print(f"构建样本列表 (双向增强: {bidirectional})...")
        
        for _, row in df.iterrows():
            p1 = str(row['protein1_uniprot_id']).strip()
            p2 = str(row['protein2_uniprot_id']).strip()
            
            p1_valid = p1 in self.valid_ids
            p2_valid = p2 in self.valid_ids
            
            # 任务 1: A -> B
            if p1_valid: 
                self.samples.append({'target_id': p1, 'binder_id': p2})
            else:
                skipped_count += 1
            
            # 任务 2: B -> A
            if bidirectional:
                if p2_valid:
                    self.samples.append({'target_id': p2, 'binder_id': p1})

        if max_samples:
            self.samples = self.samples[:max_samples]
            
        print(f"Dataset 准备就绪: {len(self.samples)} 个有效样本 (跳过缺失: {skipped_count})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        target_id = item['target_id']
        binder_id = item['binder_id']

        target_features = None
        drug_angles = None
        drug_sequence = None

        try:
            # 1. 读取 Target (从缓存)
            target_path = os.path.join(self.cache_dir, f"{target_id}.npz")
            if not os.path.exists(target_path): return None
                
            data = np.load(target_path, allow_pickle=True)
            if 'final_features' not in data: return None
            
            target_features = torch.from_numpy(data['final_features']).float()

            # =========================================================
            # [关键修改] 读取并处理分辨率数据
            # =========================================================
            resolutions = None
            
            # 如果不是消融实验，且数据中存在 resolution_score，则读取
            if not self.ablate_resolution and 'resolution_score' in data:
                resolutions = torch.from_numpy(data['resolution_score']).float()
            
            # 如果没读到（旧数据/缺失）或者开启了消融，则填充 0 (表示无信息)
            if resolutions is None:
                # 默认值设为 0 或其他您认为代表"未知/平均"的值
                resolutions = torch.zeros(target_features.shape[0], dtype=torch.float32)

            # 确保维度匹配: [SeqLen] -> [SeqLen, 1]
            if resolutions.dim() == 1:
                resolutions = resolutions.unsqueeze(-1)
            
            # 安全检查：长度必须一致
            if resolutions.shape[0] != target_features.shape[0]:
                return None
            # =========================================================

            # 2. 读取 Binder (从 PDB 实时解析)
            binder_pdb = _find_pdb_file(self.pdb_dir, binder_id)
            if not binder_pdb: return None
            
            drug_data = extract_drug_structure_features(binder_pdb)
            if drug_data is None: return None
            
            drug_sequence = sequence_to_tensor(drug_data[0])
            drug_angles = torch.tensor(drug_data[2], dtype=torch.float32)

            # 3. 最终返回
            if target_features is not None and drug_angles is not None and drug_sequence is not None:
                return {
                    'target_features': target_features,
                    'target_resolutions': resolutions,  # <--- 返回处理好的分辨率张量
                    'drug_angle_features': drug_angles,
                    'drug_sequence': drug_sequence
                }
            else:
                return None

        except Exception as e:
            # print(f"Error in __getitem__: {e}")
            return None

def collate_fn_ppi(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None

    target_feats = [b['target_features'] for b in batch]
    # =========================================================
    # [关键修改] 收集分辨率数据
    # =========================================================
    target_res = [b['target_resolutions'] for b in batch]
    
    drug_angles = [b['drug_angle_features'] for b in batch]
    drug_seqs = [b['drug_sequence'] for b in batch]

    padded_target = pad_sequence(target_feats, batch_first=True)
    
    # =========================================================
    # [关键修改] Pad 分辨率数据
    # padding_value=0 表示 Padding 区域无分辨率信息
    # =========================================================
    padded_resolutions = pad_sequence(target_res, batch_first=True, padding_value=0.0)
    
    padded_angles = pad_sequence(drug_angles, batch_first=True)
    padded_seqs = pad_sequence(drug_seqs, batch_first=True, padding_value=-1)

    B, L_tgt, _ = padded_target.shape
    context_masks = torch.zeros((B, L_tgt), dtype=torch.bool)
    for i, t in enumerate(target_feats):
        context_masks[i, t.shape[0]:] = True
        
    B, L_drug, _ = padded_angles.shape
    target_masks = torch.zeros((B, L_drug), dtype=torch.bool)
    for i, t in enumerate(drug_angles):
        target_masks[i, t.shape[0]:] = True

    return {
        'context_features': padded_target,
        'context_resolutions': padded_resolutions, # <--- 传回真实数据给 Model
        'target_angles': padded_angles,
        'target_sequences': padded_seqs,
        'context_masks': context_masks,
        'target_masks': target_masks
    }