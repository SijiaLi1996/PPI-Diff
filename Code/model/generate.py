import torch
import pandas as pd
import os
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import math

# 导入模型和数据集
from models import SequencePredictor, ConditionalAngleDiffusion, sample
from melt_dataset1 import ProteinInteractionDataset, collate_fn_ppi
# [关键] 导入刚刚创建的 NeRF 工具 (必须确保 nerf_utils.py 存在)
from nerf_utils import reconstruct_backbone_from_angles

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_pdb_file(coords, sequence, filename):
    """
    保存坐标为 PDB 文件 (仅骨架 N, CA, C)
    coords: [3*L, 3] tensor
    """
    with open(filename, 'w') as f:
        atom_idx = 1
        for i, res_name in enumerate(sequence):
            # 简单的三字母转换
            res_3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU',
                     'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
                     'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL','X':'UNK'}.get(res_name, 'UNK')
            
            if coords is not None:
                # N
                f.write(f"ATOM  {atom_idx:5d}  N   {res_3} A{i+1:4d}    {coords[3*i,0]:8.3f}{coords[3*i,1]:8.3f}{coords[3*i,2]:8.3f}  1.00  0.00           N\n")
                atom_idx += 1
                # CA
                f.write(f"ATOM  {atom_idx:5d}  CA  {res_3} A{i+1:4d}    {coords[3*i+1,0]:8.3f}{coords[3*i+1,1]:8.3f}{coords[3*i+1,2]:8.3f}  1.00  0.00           C\n")
                atom_idx += 1
                # C
                f.write(f"ATOM  {atom_idx:5d}  C   {res_3} A{i+1:4d}    {coords[3*i+2,0]:8.3f}{coords[3*i+2,1]:8.3f}{coords[3*i+2,2]:8.3f}  1.00  0.00           C\n")
                atom_idx += 1

def check_inputs(features, mask):
    if features.abs().sum() < 1e-6: return False
    if mask.all(): return False
    print(f"\n[Data Check] Shape: {features.shape} | Mean: {features.mean():.4f} | Std: {features.std():.4f}")
    return True

def generate_sequences_final(args):
    print(f"--- 设备: {DEVICE} ---")
    
    # 1. 准备输出目录
    pdb_save_dir = os.path.join(args.output_dir, "generated_pdbs")
    os.makedirs(pdb_save_dir, exist_ok=True)
    
    # 2. 加载数据集 (为了获取 context_dim 和数据)
    if not os.path.exists(args.cache_root):
        print(f"⚠️ 警告: 缓存目录 {args.cache_root} 不存在。")
    
    dataset = ProteinInteractionDataset(
        links_file=args.links_file,
        features_raw_dir=args.features_dir,
        pdb_dir=args.pdb_dir,
        cache_root=args.cache_root,
        score_threshold=args.score_threshold,
        max_samples=args.max_samples,
        sep='\t',
        use_esm=True,
        ablate_resolution=False,
        bidirectional=False
    )
    
    if len(dataset) == 0: return
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_ppi, shuffle=False)

    # 自动检测 Context Dimension
    sample_item = None
    for i in range(min(10, len(dataset))):
        if dataset[i] is not None:
            sample_item = dataset[i]
            break
    if sample_item is not None:
        args.context_dim = sample_item['target_features'].shape[-1]
        print(f"✅ 自动检测 Context Dimension: {args.context_dim}")

    # =================================================================================
    # [关键修改] 3. 智能加载 Checkpoint 参数
    # 先加载文件，看看里面有没有存 'args'，如果有，就覆盖当前的设置
    # =================================================================================
    print(f"📥 正在读取 Checkpoint 文件: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE)
    
    if 'args' in ckpt:
        train_args = ckpt['args']
        print("----------------------------------------------------------------")
        print("⚡️ 智能加载: 检测到 Checkpoint 中保存了训练参数，正在覆盖默认设置...")
        
        # 覆盖关键的模型结构参数
        if hasattr(train_args, 'num_layers'): args.num_layers = train_args.num_layers
        if hasattr(train_args, 'hidden_dim'): args.hidden_dim = train_args.hidden_dim
        if hasattr(train_args, 'angle_dim'): args.angle_dim = train_args.angle_dim
        if hasattr(train_args, 'vocab_size'): args.vocab_size = train_args.vocab_size
        
        print(f"   -> 更新后 Num Layers: {args.num_layers}")
        print(f"   -> 更新后 Hidden Dim: {args.hidden_dim}")
        print("----------------------------------------------------------------")
    else:
        print("⚠️ Checkpoint 中未找到训练参数，将使用命令行参数初始化模型。")

    # 4. 初始化模型 (使用更新后的 args)
    print(f"🧠 初始化模型 (Layers={args.num_layers}, Hidden={args.hidden_dim})...")
    
    angle_model = ConditionalAngleDiffusion(
        angle_dim=args.angle_dim, context_dim=args.context_dim, hidden_dim=args.hidden_dim,
        timesteps=args.timesteps, num_layers=args.num_layers
    ).to(DEVICE)
    
    seq_model = SequencePredictor(
        angle_dim=args.angle_dim, context_dim=args.context_dim, hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size, num_layers=args.num_layers
    ).to(DEVICE)

    # 5. 加载权重 State Dict
    if 'angle_model_state_dict' in ckpt:
        angle_model.load_state_dict(ckpt['angle_model_state_dict'])
        seq_model.load_state_dict(ckpt['seq_model_state_dict'])
        print("✅ 权重加载成功 (Standard Format)")
    else:
        # 兼容旧版本保存方式
        print("⚠️ 检测到旧版 Checkpoint 格式，尝试直接加载 Angle Model...")
        angle_model.load_state_dict(ckpt)
    
    angle_model.eval()
    seq_model.eval()

    # 6. 生成循环
    aa_vocab = "ARNDCQEGHILKMFPSTWYV"
    idx_to_aa = {i: aa for i, aa in enumerate(aa_vocab)}
    results = []

    print("\n🚀 开始生成并保存 PDB...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if batch is None: continue
            
            ctx_feat = batch['context_features'].to(DEVICE)
            ctx_res = batch['context_resolutions'].to(DEVICE)
            raw_mask = batch['context_masks'].to(DEVICE)
            tgt_seq_gt = batch['target_sequences'].to(DEVICE)
            
            # 处理 Mask (True为忽略)
            mask = ~raw_mask.bool() if raw_mask.float().mean() > 0.5 else raw_mask.bool()
            
            if i == 0: check_inputs(ctx_feat, mask)
            
            seq_len = batch['target_angles'].shape[1]

            # --- A. 生成角度 ---
            gen_ang = sample(
                model=angle_model, context=ctx_feat, resolution=ctx_res, 
                sequence_length=seq_len, batch_size=1, timesteps=1000, 
                context_mask=mask
            )
            
            # --- B. 序列预测 ---
            logits = seq_model(
                x_angles=gen_ang, context=ctx_feat, resolution=ctx_res, memory_key_padding_mask=mask 
            )
            
            # --- C. 序列解码 (固定 Temperature=0.1) ---
            T = 0.1
            scaled_logits = logits / T
            probs = torch.softmax(scaled_logits, dim=-1)
            pred_indices = torch.multinomial(probs.view(-1, args.vocab_size), num_samples=1).view(1, -1).cpu().numpy()[0]
            pred_seq = "".join([idx_to_aa.get(idx, 'X') for idx in pred_indices])
            
            # 获取 GT 序列
            gt_indices = tgt_seq_gt.cpu().numpy()[0]
            gt_seq = "".join([idx_to_aa.get(idx, '') for idx in gt_indices if idx in idx_to_aa])

            # --- D. 坐标重建与保存 ---
            try:
                # 使用 nerf_utils 重建坐标 [SeqLen, 12] -> [3*SeqLen, 3]
                coords_3d = reconstruct_backbone_from_angles(gen_ang.squeeze(0))
                
                pdb_filename = os.path.join(pdb_save_dir, f"sample_{i}.pdb")
                save_pdb_file(coords_3d, pred_seq, pdb_filename)
            except Exception as e:
                print(f"⚠️ 样本 {i} 坐标重建失败: {e}")

            # 保存结果到列表
            results.append({
                "id": f"sample_{i}",
                "temperature": T,
                "generated_sequence": pred_seq,
                "gt_sequence": gt_seq
            })

    # 保存 CSV
    if hasattr(args, 'output_dir'):
        os.makedirs(args.output_dir, exist_ok=True)
        df = pd.DataFrame(results)
        save_path = os.path.join(args.output_dir, "generation_results.csv")
        df.to_csv(save_path, index=False)
        print(f"✅ CSV 结果已保存: {save_path}")
        print(f"✅ PDB 结构已保存至: {pdb_save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 路径参数
    parser.add_argument('--links_file', default="/home/featurize/work/protein/human_protein_interactions_verified_200.tsv")
    parser.add_argument('--features_dir', default="/home/featurize/work/protein/protein_features_by_ppi200")
    parser.add_argument('--pdb_dir', default="/home/featurize/work/protein/ppi_pdb_by_uniprot")
    parser.add_argument('--cache_root', default="./cached_ppi_data") 
    # [修改] 修复了路径中的中文句号，并建议检查文件名
    parser.add_argument('--checkpoint', default="./saved_models/full_model_v1/checkpoint_epoch_500.pth")
    parser.add_argument('--output_dir', default="./final_results")
    
    parser.add_argument('--max_samples', type=int, default=100)
    parser.add_argument('--score_threshold', type=int, default=400)
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--context_dim', type=int, default=334)
    parser.add_argument('--angle_dim', type=int, default=12)
    # [修改] 将默认 num_layers 改为 4，以匹配你训练时的设置
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--vocab_size', type=int, default=25)

    args = parser.parse_args()
    generate_sequences_final(args)