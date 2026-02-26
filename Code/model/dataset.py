# dataset.py (根据实际数据结构修改)
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

class ProteinInteractionDataset(Dataset):
    """
    蛋白质相互作用数据集类 - 适配实际数据结构
    """
    def __init__(self, links_file, features_dir, score_threshold=400, max_samples=None, sep='\t'):
        self.features_dir = features_dir
        self.score_threshold = score_threshold
        self.max_seq_len = 512
        
        # 氨基酸到索引的映射
        self.aa_to_idx = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }
        
        # 根据检查结果设置键名
        self.angle_key = 'angle_features'
        self.feature_key = 'fused_features'
        self.sequence_key = 'sequence'
        self.resolution_key = 'resolution'
        
        logging.info(f"使用数据键映射:")
        logging.info(f"  角度键: {self.angle_key}")
        logging.info(f"  特征键: {self.feature_key}")
        logging.info(f"  序列键: {self.sequence_key}")
        logging.info(f"  分辨率键: {self.resolution_key}")
        
        # 加载相互作用对
        self.interaction_pairs = self._load_interaction_pairs(links_file, sep)
        
        # 限制样本数量
        if max_samples and max_samples < len(self.interaction_pairs):
            self.interaction_pairs = self.interaction_pairs[:max_samples]
            logging.info(f"样本数已从 {len(self.interaction_pairs)} 截断至 {max_samples}")

        logging.info(f"数据集初始化完成。共生成 {len(self.interaction_pairs)} 个训练样本")

    def _find_feature_file(self, uniprot_id):
        """在UniProt ID文件夹下找到第一个 .npz 特征文件"""
        folder_path = os.path.join(self.features_dir, uniprot_id)
        if not os.path.isdir(folder_path):
            return None
        for f in os.listdir(folder_path):
            if f.endswith('.npz'):
                return os.path.join(folder_path, f)
        return None

    def _load_interaction_pairs(self, links_file, sep):
        """加载蛋白质相互作用对"""
        logging.info(f"正在从 {links_file} 加载相互作用关系...")
        
        try:
            df_links = pd.read_csv(links_file, sep=sep)
            logging.info(f"成功读取文件，形状: {df_links.shape}")
            
        except Exception as e:
            logging.error(f"读取文件 '{links_file}' 时出错: {e}")
            return []

        # 检查必要的列
        required_cols = ['protein1_uniprot_id', 'protein2_uniprot_id', 'combined_score']
        missing_cols = [col for col in required_cols if col not in df_links.columns]
        if missing_cols:
            logging.error(f"文件缺少必要的列: {missing_cols}")
            return []

        # 按分数过滤
        df_filtered = df_links[df_links['combined_score'] >= self.score_threshold]
        logging.info(f"筛选后 (分数 >= {self.score_threshold}): {len(df_filtered)} 条相互作用")

        valid_pairs = []
        
        for _, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="检查特征文件"):
            p1_id = str(row['protein1_uniprot_id']).strip()
            p2_id = str(row['protein2_uniprot_id']).strip()
            
            p1_npz = self._find_feature_file(p1_id)
            p2_npz = self._find_feature_file(p2_id)

            if p1_npz and p2_npz:
                # 验证文件是否包含必要的键
                if self._validate_npz_file(p1_npz) and self._validate_npz_file(p2_npz):
                    valid_pairs.append({
                        'target_npz': p1_npz,
                        'context_npz': p2_npz,
                        'target_id': p1_id,
                        'context_id': p2_id
                    })
                    valid_pairs.append({
                        'target_npz': p2_npz,
                        'context_npz': p1_npz,
                        'target_id': p2_id,
                        'context_id': p1_id
                    })
        
        logging.info(f"找到 {len(valid_pairs)} 个有效的训练样本")
        return valid_pairs

    def _validate_npz_file(self, npz_path):
        """验证npz文件是否包含必要的数据"""
        try:
            data = np.load(npz_path, allow_pickle=True)
            keys = set(data.keys())
            
            # 检查必要的键是否存在
            required_keys = [self.angle_key, self.feature_key]
            return all(key in keys for key in required_keys)
            
        except Exception as e:
            logging.warning(f"验证文件 {npz_path} 时出错: {e}")
            return False

    def __len__(self):
        return len(self.interaction_pairs)

    def __getitem__(self, idx):
        """获取单个样本"""
        try:
            if idx >= len(self.interaction_pairs):
                return None
                
            sample_info = self.interaction_pairs[idx]
            target_npz = sample_info['target_npz']
            context_npz = sample_info['context_npz']
            
            # 加载特征文件
            target_feat = np.load(target_npz, allow_pickle=True)
            context_feat = np.load(context_npz, allow_pickle=True)
            
            # 提取目标角度数据
            target_angles = self._extract_angles(target_feat)
            if target_angles is None:
                return None
            
            # 提取上下文特征数据
            context_features = self._extract_features(context_feat)
            if context_features is None:
                return None
            
            # 处理分辨率数据
            context_resolutions = self._extract_resolution(context_feat, len(context_features))
            
            # 处理序列数据
            target_sequences = self._extract_sequence(target_feat, len(target_angles))
            
            # 确保长度一致
            min_target_len = min(len(target_angles), len(target_sequences))
            target_angles = target_angles[:min_target_len]
            target_sequences = target_sequences[:min_target_len]
            
            min_context_len = min(len(context_features), len(context_resolutions))
            context_features = context_features[:min_context_len]
            context_resolutions = context_resolutions[:min_context_len]
            
            # 应用长度限制
            if len(target_angles) > self.max_seq_len:
                target_angles = target_angles[:self.max_seq_len]
                target_sequences = target_sequences[:self.max_seq_len]
            
            if len(context_features) > self.max_seq_len:
                context_features = context_features[:self.max_seq_len]
                context_resolutions = context_resolutions[:self.max_seq_len]
            
            return {
                'target_angles': target_angles,
                'context_features': context_features,
                'context_resolutions': context_resolutions,
                'target_sequences': target_sequences
            }
            
        except Exception as e:
            logging.error(f"加载数据时出错 (idx={idx}): {str(e)}")
            return None

    def _extract_angles(self, feat_data):
        """提取角度数据"""
        try:
            if self.angle_key in feat_data:
                angles = feat_data[self.angle_key]
                if isinstance(angles, np.ndarray) and angles.size > 0:
                    # 确保是2D数组，形状为 (seq_len, 12)
                    if angles.ndim == 1:
                        angles = angles.reshape(-1, 1)
                    elif angles.ndim == 2 and angles.shape[1] != 12:
                        # 如果不是12维，调整到12维
                        seq_len = angles.shape[0]
                        if angles.shape[1] < 12:
                            # 填充到12维
                            padding = np.zeros((seq_len, 12 - angles.shape[1]))
                            angles = np.concatenate([angles, padding], axis=1)
                        else:
                            # 截断到12维
                            angles = angles[:, :12]
                    
                    return torch.tensor(angles, dtype=torch.float32)
            
            logging.warning(f"无法找到角度数据键: {self.angle_key}")
            return None
            
        except Exception as e:
            logging.error(f"提取角度数据时出错: {str(e)}")
            return None

    def _extract_features(self, feat_data):
        """提取特征数据"""
        try:
            if self.feature_key in feat_data:
                features = feat_data[self.feature_key]
                if isinstance(features, np.ndarray) and features.size > 0:
                    return torch.tensor(features, dtype=torch.float32)
            
            logging.warning(f"无法找到特征数据键: {self.feature_key}")
            return None
            
        except Exception as e:
            logging.error(f"提取特征数据时出错: {str(e)}")
            return None

    def _extract_resolution(self, feat_data, target_length):
        """提取分辨率数据"""
        try:
            if self.resolution_key in feat_data:
                resolution = feat_data[self.resolution_key]
                
                # 处理标量分辨率
                if isinstance(resolution, (np.ndarray, np.number)) and np.isscalar(resolution):
                    res_value = float(resolution)
                    return torch.full((target_length,), res_value, dtype=torch.float32)
                elif isinstance(resolution, np.ndarray):
                    if resolution.ndim == 0:  # 0维数组（标量）
                        res_value = float(resolution.item())
                        return torch.full((target_length,), res_value, dtype=torch.float32)
                    else:
                        # 数组形式的分辨率
                        res_tensor = torch.tensor(resolution, dtype=torch.float32)
                        if len(res_tensor) == target_length:
                            return res_tensor
                        elif len(res_tensor) == 1:
                            return res_tensor.repeat(target_length)
                        else:
                            # 调整长度
                            if len(res_tensor) > target_length:
                                return res_tensor[:target_length]
                            else:
                                last_val = res_tensor[-1] if len(res_tensor) > 0 else 2.0
                                padding = torch.full((target_length - len(res_tensor),), last_val)
                                return torch.cat([res_tensor, padding])
                else:
                    # 直接转换为浮点数
                    res_value = float(resolution)
                    return torch.full((target_length,), res_value, dtype=torch.float32)
        except Exception as e:
            logging.warning(f"提取分辨率数据时出错: {str(e)}")
        
        # 使用默认分辨率
        return torch.full((target_length,), 2.0, dtype=torch.float32)

    def _extract_sequence(self, feat_data, target_length):
        """提取序列数据"""
        try:
            if self.sequence_key in feat_data:
                sequence = feat_data[self.sequence_key]
                
                # 处理字符串序列
                if isinstance(sequence, (str, np.str_)):
                    seq_str = str(sequence)
                    indices = [self.aa_to_idx.get(aa.upper(), 0) for aa in seq_str]
                    seq_tensor = torch.tensor(indices, dtype=torch.long)
                elif isinstance(sequence, np.ndarray):
                    if sequence.ndim == 0:  # 0维数组（标量字符串）
                        seq_str = str(sequence.item())
                        indices = [self.aa_to_idx.get(aa.upper(), 0) for aa in seq_str]
                        seq_tensor = torch.tensor(indices, dtype=torch.long)
                    elif sequence.dtype.kind in ['U', 'S']:  # 字符串数组
                        if sequence.ndim == 1:
                            # 字符数组
                            indices = [self.aa_to_idx.get(str(aa).upper(), 0) for aa in sequence]
                            seq_tensor = torch.tensor(indices, dtype=torch.long)
                        else:
                            # 多维字符串数组，取第一个
                            seq_str = str(sequence.flat[0])
                            indices = [self.aa_to_idx.get(aa.upper(), 0) for aa in seq_str]
                            seq_tensor = torch.tensor(indices, dtype=torch.long)
                    else:
                        # 数值数组
                        seq_tensor = torch.tensor(sequence, dtype=torch.long)
                else:
                    # 其他类型，尝试转换为字符串
                    seq_str = str(sequence)
                    indices = [self.aa_to_idx.get(aa.upper(), 0) for aa in seq_str]
                    seq_tensor = torch.tensor(indices, dtype=torch.long)
                
                # 调整长度
                if len(seq_tensor) == target_length:
                    return seq_tensor
                elif len(seq_tensor) > target_length:
                    return seq_tensor[:target_length]
                else:
                    # 用0填充
                    padding = torch.zeros(target_length - len(seq_tensor), dtype=torch.long)
                    return torch.cat([seq_tensor, padding])
                    
        except Exception as e:
            logging.warning(f"提取序列数据时出错: {str(e)}")
        
        # 使用默认序列
        return torch.zeros(target_length, dtype=torch.long)


def collate_fn_ppi(batch):
    """修复的批次整理函数"""
    if not batch:
        return None
    
    # 过滤掉None样本
    valid_batch = []
    required_keys = ['target_angles', 'context_features', 'context_resolutions', 'target_sequences']
    
    for item in batch:
        if item is not None and isinstance(item, dict):
            if all(key in item for key in required_keys):
                # 检查张量是否有效
                try:
                    target_angles = item['target_angles']
                    context_features = item['context_features']
                    
                    # 确保张量至少有一个元素
                    if (isinstance(target_angles, torch.Tensor) and target_angles.numel() > 0 and
                        isinstance(context_features, torch.Tensor) and context_features.numel() > 0):
                        valid_batch.append(item)
                    else:
                        logging.warning(f"跳过空张量的数据项")
                except Exception as e:
                    logging.warning(f"检查数据项时出错: {e}")
            else:
                missing_keys = [key for key in required_keys if key not in item]
                logging.warning(f"数据项缺少键: {missing_keys}")
    
    if not valid_batch:
        logging.warning("批次中没有有效数据")
        return None
    
    batch = valid_batch
    MAX_SEQ_LEN = 512
    
    try:
        # 计算最大长度时添加安全检查
        target_lengths = []
        context_lengths = []
        
        for item in batch:
            target_len = len(item['target_angles'])
            context_len = len(item['context_features'])
            
            # 确保长度合理
            if target_len > 0 and context_len > 0:
                target_lengths.append(min(target_len, MAX_SEQ_LEN))
                context_lengths.append(min(context_len, MAX_SEQ_LEN))
        
        if not target_lengths or not context_lengths:
            logging.error("无法计算有效的序列长度")
            return None
        
        max_target_len = max(target_lengths)
        max_context_len = max(context_lengths)
        
        # 确保最小长度
        max_target_len = max(max_target_len, 1)
        max_context_len = max(max_context_len, 1)
        
        logging.debug(f"批次大小: {len(batch)}, 目标最大长度: {max_target_len}, 上下文最大长度: {max_context_len}")
        
        # 准备列表
        target_angles_list = []
        context_features_list = []
        context_resolutions_list = []
        target_masks_list = []
        context_masks_list = []
        target_sequences_list = []
        
        for idx, item in enumerate(batch):
            try:
                # 处理目标角度
                target_angles = item['target_angles']
                orig_target_len = len(target_angles)
                
                if orig_target_len == 0:
                    logging.warning(f"批次项 {idx} 的目标角度为空，跳过")
                    continue
                
                if orig_target_len > max_target_len:
                    target_angles = target_angles[:max_target_len]
                    actual_target_len = max_target_len
                else:
                    actual_target_len = orig_target_len
                    if orig_target_len < max_target_len:
                        # 确保填充维度正确
                        if len(target_angles.shape) == 2:
                            padding = torch.zeros(max_target_len - orig_target_len, target_angles.shape[1])
                        else:
                            padding = torch.zeros(max_target_len - orig_target_len)
                        target_angles = torch.cat([target_angles, padding], dim=0)
                
                target_angles_list.append(target_angles)
                
                # 处理上下文特征
                context_features = item['context_features']
                orig_context_len = len(context_features)
                
                if orig_context_len == 0:
                    logging.warning(f"批次项 {idx} 的上下文特征为空，跳过")
                    continue
                
                if orig_context_len > max_context_len:
                    context_features = context_features[:max_context_len]
                    actual_context_len = max_context_len
                else:
                    actual_context_len = orig_context_len
                    if orig_context_len < max_context_len:
                        # 确保填充维度正确
                        if len(context_features.shape) == 2:
                            padding = torch.zeros(max_context_len - orig_context_len, context_features.shape[1])
                        else:
                            padding = torch.zeros(max_context_len - orig_context_len)
                        context_features = torch.cat([context_features, padding], dim=0)
                
                context_features_list.append(context_features)
                
                # 处理分辨率
                context_resolutions = item['context_resolutions']
                if len(context_resolutions) > max_context_len:
                    context_resolutions = context_resolutions[:max_context_len]
                elif len(context_resolutions) < max_context_len:
                    fill_value = context_resolutions[-1] if len(context_resolutions) > 0 else 2.0
                    padding = torch.full((max_context_len - len(context_resolutions),), fill_value)
                    context_resolutions = torch.cat([context_resolutions, padding], dim=0)
                context_resolutions_list.append(context_resolutions)
                
                # 处理序列
                target_sequences = item['target_sequences']
                if len(target_sequences) > max_target_len:
                    target_sequences = target_sequences[:max_target_len]
                elif len(target_sequences) < max_target_len:
                    padding = torch.full((max_target_len - len(target_sequences),), -1)
                    target_sequences = torch.cat([target_sequences, padding], dim=0)
                target_sequences_list.append(target_sequences)
                
                # 创建掩码 (True表示填充位置，应该被忽略)
                target_mask = torch.zeros(max_target_len, dtype=torch.bool)
                if actual_target_len < max_target_len:
                    target_mask[actual_target_len:] = True
                target_masks_list.append(target_mask)
                
                context_mask = torch.zeros(max_context_len, dtype=torch.bool)
                if actual_context_len < max_context_len:
                    context_mask[actual_context_len:] = True
                context_masks_list.append(context_mask)
                
            except Exception as e:
                logging.error(f"处理批次项 {idx} 时出错: {e}")
                continue
        
        # 检查是否有有效的数据
        if not target_angles_list or not context_features_list:
            logging.error("没有有效的批次数据")
            return None
        
        # 确保所有列表长度一致
        min_batch_size = min(len(target_angles_list), len(context_features_list), 
                            len(context_resolutions_list), len(target_masks_list),
                            len(context_masks_list), len(target_sequences_list))
        
        if min_batch_size == 0:
            logging.error("批次大小为0")
            return None
        
        # 截断到最小大小
        target_angles_list = target_angles_list[:min_batch_size]
        context_features_list = context_features_list[:min_batch_size]
        context_resolutions_list = context_resolutions_list[:min_batch_size]
        target_masks_list = target_masks_list[:min_batch_size]
        context_masks_list = context_masks_list[:min_batch_size]
        target_sequences_list = target_sequences_list[:min_batch_size]
        
        # 堆叠张量
        result = {
            'target_angles': torch.stack(target_angles_list),
            'context_features': torch.stack(context_features_list),
            'context_resolutions': torch.stack(context_resolutions_list),
            'target_masks': torch.stack(target_masks_list),
            'context_masks': torch.stack(context_masks_list),
            'target_sequences': torch.stack(target_sequences_list)
        }
        
        # 验证结果
        logging.debug(f"最终批次形状:")
        for key, tensor in result.items():
            logging.debug(f"  {key}: {tensor.shape}")
        
        return result
        
    except Exception as e:
        logging.error(f"批次整理时出错: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None