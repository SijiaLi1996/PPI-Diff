# utils.py
import os
import torch

def get_device():
    """获取当前可用的设备 (CPU 或 CUDA)。"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def _find_protein_feature_files(protein_feature_base_dir, uniprot_id):
    """在指定的 UniProt ID 子文件夹中查找所有特征文件 (.npz)。"""
    folder_path = os.path.join(protein_feature_base_dir, uniprot_id)
    if not os.path.isdir(folder_path):
        return []
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('_features.npz') or f.endswith('.npz')]

def _find_pdb_file(pdb_base_dir, id_code):
    """在指定的 ID 子文件夹中查找 PDB 文件。"""
    folder_path = os.path.join(pdb_base_dir, id_code)
    if not os.path.isdir(folder_path):
        return None
    for filename in os.listdir(folder_path):
        if filename.startswith('filtered_') and filename.endswith('.pdb'):
            return os.path.join(folder_path, filename)
    return None

# 您也可以把 NUM_AMINO_ACIDS 等常量放在这里，如果它被多处使用
# 但目前它只在 drug_structure 和 models 中使用，所以保持原样也可以