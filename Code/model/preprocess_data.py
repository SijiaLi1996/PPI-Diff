# preprocess_data.py
import os
import torch
from tqdm import tqdm
import pandas as pd
import argparse

# 导入您已有的数据处理函数
from drug_structure import extract_drug_structure_features, sequence_to_tensor
from fuse_features import fuse_target_features
# --- 修正：从 utils.py 导入，打破循环 ---
from utils import _find_pdb_file, _find_protein_feature_files

def preprocess_and_save(args):
    """
    遍历所有数据，进行预处理，并保存为 .pt 文件。
    """
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"预处理数据将保存到: {args.output_dir}")

    # 加载关系文件
    df = pd.read_csv(args.relations_tsv, sep='\t')
    
    processed_count = 0
    # 为了避免重复处理，使用集合来跟踪已处理的对
    processed_pairs = set()

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="预处理数据"):
        prot1_id = str(row['protein1_uniprot_id']).strip()
        prot2_id = str(row['protein2_uniprot_id']).strip()

        # --- 处理 A -> B ---
        pair_key_ab = (prot1_id, prot2_id)
        if pair_key_ab not in processed_pairs:
            try:
                prot1_pdb = _find_pdb_file(args.pdb_dir, prot1_id)
                prot2_feats = _find_protein_feature_files(args.features_dir, prot2_id)

                if prot1_pdb and prot2_feats:
                    drug_data = extract_drug_structure_features(prot1_pdb)
                    target_data = fuse_target_features(prot2_feats, prot2_id, use_esm=True, use_transformer=True)

                    if drug_data and target_data and target_data.get('final_features') is not None:
                        sample = {
                            'drug_sequence': sequence_to_tensor(drug_data[0]),
                            'drug_angle_features': torch.tensor(drug_data[2], dtype=torch.float32),
                            'target_features': target_data['final_features'].clone().detach(),
                        }
                        torch.save(sample, os.path.join(args.output_dir, f"sample_{processed_count}.pt"))
                        processed_count += 1
                        processed_pairs.add(pair_key_ab)
            except Exception:
                continue

        # --- 处理 B -> A ---
        pair_key_ba = (prot2_id, prot1_id)
        if pair_key_ba not in processed_pairs:
            try:
                prot2_pdb = _find_pdb_file(args.pdb_dir, prot2_id)
                prot1_feats = _find_protein_feature_files(args.features_dir, prot1_id)

                if prot2_pdb and prot1_feats:
                    drug_data = extract_drug_structure_features(prot2_pdb)
                    target_data = fuse_target_features(prot1_feats, prot1_id, use_esm=True, use_transformer=True)

                    if drug_data and target_data and target_data.get('final_features') is not None:
                        sample = {
                            'drug_sequence': sequence_to_tensor(drug_data[0]),
                            'drug_angle_features': torch.tensor(drug_data[2], dtype=torch.float32),
                            'target_features': target_data['final_features'].clone().detach(),
                        }
                        torch.save(sample, os.path.join(args.output_dir, f"sample_{processed_count}.pt"))
                        processed_count += 1
                        processed_pairs.add(pair_key_ba)
            except Exception:
                continue

    print(f"预处理完成！共成功处理并保存了 {processed_count} 个样本。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="数据预处理脚本")
    parser.add_argument('--relations_tsv', type=str, default="/home/featurize/work/protein/human_protein_interactions_verified_200.tsv")
    parser.add_argument('--pdb_dir', type=str, default="/home/featurize/work/protein/ppi_pdb_by_uniprot")
    parser.add_argument('--features_dir', type=str, default="/home/featurize/work/protein/protein_features_by_ppi200")
    parser.add_argument('--output_dir', type=str, default="./preprocessed_data_ppi")
    args = parser.parse_args()
    preprocess_and_save(args)