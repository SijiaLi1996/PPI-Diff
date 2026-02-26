import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import logging
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ==============================================================================
# --- 导入自定义模块 ---
# 请确保 melt_dataset1.py, models.py, utils.py 在同一目录下
# ==============================================================================
try:
    from models import ConditionalAngleDiffusion, SequencePredictor
    from melt_dataset1 import ProteinInteractionDataset, collate_fn_ppi
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保 'models.py' 和 'melt_dataset1.py' 存在于当前目录中。")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# --- 扩散模型辅助函数 (Schedule) ---
# ==============================================================================
def get_ddpm_schedule(timesteps, device):
    beta_start, beta_end = 0.0001, 0.02
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return torch.sqrt(alphas_cumprod), torch.sqrt(1. - alphas_cumprod)

def q_sample(x_start, t, sqrt_alphas, sqrt_one_minus_alphas, noise=None):
    if noise is None: noise = torch.randn_like(x_start)
    sqrt_alphas_t = sqrt_alphas[t].reshape(-1, 1, 1)
    sqrt_one_minus_alphas_t = sqrt_one_minus_alphas[t].reshape(-1, 1, 1)
    return sqrt_alphas_t * x_start + sqrt_one_minus_alphas_t * noise

# ==============================================================================
# --- 日志设置 ---
# ==============================================================================
def setup_logger(save_dir):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, force=True)
    logger = logging.getLogger()
    
    if logger.hasHandlers(): logger.handlers.clear()
    
    os.makedirs(save_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(save_dir, "train_log.txt"))
    file_handler.setFormatter(logging.Formatter(log_format))
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# ==============================================================================
# --- 主训练函数 ---
# ==============================================================================
def train(args):
    # --- 1. 确定实验名称与保存路径 ---
    # 根据参数自动决定实验名称，方便管理
    exp_parts = ["model"]
    
    if not args.use_esm:
        exp_parts.append("no_esm")
    if args.ablate_resolution:
        exp_parts.append("no_res") # 消融分辨率
    if not args.bidirectional:
        exp_parts.append("uni_dir")
        
    if len(exp_parts) == 1:
        exp_name = "full_model_v1" # 完整模型
    else:
        exp_name = "ablation_" + "_".join(exp_parts[1:])

    save_dir = os.path.join(args.save_root, exp_name)
    
    logger = setup_logger(save_dir)
    logger.info(f"=== 启动训练任务: {exp_name} ===")
    logger.info(f"📂 模型保存路径: {save_dir}")
    logger.info(f"⚙️ 参数: Epochs={args.epochs}, Batch={args.batch_size}, ESM={args.use_esm}, ResAware={not args.ablate_resolution}")
    
    # --- 2. 初始化数据集 ---
    dataset = ProteinInteractionDataset(
        links_file=args.links_file,
        features_raw_dir=args.features_dir,
        pdb_dir=args.pdb_dir,
        cache_root=args.cache_root,
        score_threshold=args.score_threshold,
        use_esm=args.use_esm,
        ablate_resolution=args.ablate_resolution, # 传递消融参数
        bidirectional=args.bidirectional
    )
    
    if len(dataset) == 0:
        logger.error("❌ 数据集为空，请检查路径或 Score 阈值。")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_fn_ppi, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True
    )

    # --- 3. 自动检测维度 ---
    # 尝试获取一个样本以确定 Context 维度 (因为 ESM 会改变维度)
    sample = None
    for i in range(min(50, len(dataset))):
        s = dataset[i]
        if s is not None: 
            sample = s
            break
        
    if sample is None:
        logger.error("❌ 无法获取任何有效样本来检测维度。")
        return
        
    real_context_dim = sample['target_features'].shape[-1]
    logger.info(f"📏 检测到 Context 特征维度: {real_context_dim}")
    args.context_dim = real_context_dim

    # --- 4. 初始化模型 ---
    angle_model = ConditionalAngleDiffusion(
        angle_dim=args.angle_dim, context_dim=args.context_dim, hidden_dim=args.hidden_dim,
        timesteps=args.timesteps, num_layers=args.num_layers
    ).to(DEVICE)
    
    seq_model = SequencePredictor(
        angle_dim=args.angle_dim, context_dim=args.context_dim, hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size, num_layers=args.num_layers
    ).to(DEVICE)

    # 优化器与调度器
    optimizer = torch.optim.AdamW(list(angle_model.parameters()) + list(seq_model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    # DDPM 参数
    sqrt_alphas, sqrt_one_minus_alphas = get_ddpm_schedule(args.timesteps, DEVICE)

    # --- 5. 断点续训逻辑 ---
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"📥 正在加载 Checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=DEVICE)
            
            angle_model.load_state_dict(checkpoint['angle_model_state_dict'])
            seq_model.load_state_dict(checkpoint['seq_model_state_dict'])
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint['epoch']
            logger.info(f"✅ 成功恢复！从 Epoch {start_epoch + 1} 继续。")
            
            # 同步 Scheduler
            for _ in range(start_epoch):
                scheduler.step()
        else:
            logger.error(f"❌ Checkpoint 文件不存在: {args.resume}")
            return

    # --- 6. 训练循环 ---
    for epoch in range(start_epoch, args.epochs):
        angle_model.train()
        seq_model.train()
        
        epoch_loss = 0.0
        angle_loss_sum = 0.0
        seq_loss_sum = 0.0
        valid_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for batch in pbar:
            if batch is None: continue
            
            # 数据上 GPU
            tgt_angles = batch['target_angles'].to(DEVICE, non_blocking=True)
            ctx_feats = batch['context_features'].to(DEVICE, non_blocking=True)
            
            # [关键] 获取分辨率数据
            ctx_res = batch['context_resolutions'].to(DEVICE, non_blocking=True)
            
            tgt_mask = batch['target_masks'].to(DEVICE, non_blocking=True)
            ctx_mask = batch['context_masks'].to(DEVICE, non_blocking=True)
            tgt_seq = batch['target_sequences'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                # --- A. 角度扩散损失 ---
                t = torch.randint(0, args.timesteps, (tgt_angles.size(0),), device=DEVICE).long()
                noise = torch.randn_like(tgt_angles)
                x_noisy = q_sample(tgt_angles, t, sqrt_alphas, sqrt_one_minus_alphas, noise)
                
                # 传入 ctx_res (分辨率)
                pred_noise = angle_model(x_noisy, ctx_feats, t, ctx_res, tgt_mask, ctx_mask)
                loss_ang = F.mse_loss(pred_noise[~tgt_mask], noise[~tgt_mask])
                
                # --- B. 序列预测损失 ---
                # 传入 ctx_res (分辨率)
                pred_logits = seq_model(tgt_angles, ctx_feats, ctx_res, tgt_mask, ctx_mask)
                logits_flat = pred_logits.view(-1, args.vocab_size)
                targets_flat = tgt_seq.view(-1)
                
                valid_idx = targets_flat != -1
                if valid_idx.sum() > 0:
                    loss_seq = F.cross_entropy(logits_flat[valid_idx], targets_flat[valid_idx])
                else:
                    loss_seq = torch.tensor(0.0, device=DEVICE)
                
                # 总损失
                total_loss = args.angle_weight * loss_ang + args.seq_weight * loss_seq

            # 反向传播
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(angle_model.parameters()) + list(seq_model.parameters()), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            # 统计
            bs = tgt_angles.size(0)
            epoch_loss += total_loss.item() * bs
            angle_loss_sum += loss_ang.item() * bs
            seq_loss_sum += loss_seq.item() * bs
            valid_batches += bs
            
            pbar.set_postfix({'L': f"{total_loss.item():.4f}"})

        scheduler.step()
        
        # --- Epoch 结束日志与保存 ---
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            avg_ang = angle_loss_sum / valid_batches
            avg_seq = seq_loss_sum / valid_batches
            
            logger.info(f"Epoch {epoch+1:03d} | Total: {avg_loss:.4f} | Ang: {avg_ang:.4f} | Seq: {avg_seq:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # 保存快照 (每 save_interval 保存一次，且保存最后一个)
            if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                save_filename = f'checkpoint_epoch_{epoch+1}.pth'
                save_path = os.path.join(save_dir, save_filename)
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'angle_model_state_dict': angle_model.state_dict(),
                    'seq_model_state_dict': seq_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args) # 保存参数配置
                }
                torch.save(checkpoint, save_path)
                logger.info(f"💾 模型快照已保存: {save_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # --- 路径配置 (请修改为你的实际路径) ---
    parser.add_argument('--links_file', default="./human_protein_interactions_verified_200.tsv", help="PPI 关系列表 TSV")
    parser.add_argument('--features_dir', default="./features_raw", help="原始 .npz 特征目录")
    parser.add_argument('--pdb_dir', default="./ppi_pdb_by_uniprot", help="PDB 文件目录")
    parser.add_argument('--cache_root', default="./cached_ppi_data", help="处理后数据的缓存目录")
    parser.add_argument('--save_root', default="./saved_models", help="模型保存根目录")
    
    # --- 训练核心参数 (默认 500 轮, 50 间隔) ---
    parser.add_argument('--epochs', type=int, default=500, help="总训练轮数")
    parser.add_argument('--save_interval', type=int, default=50, help="保存快照的间隔")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None, help="断点续训路径")
    
    # --- 模型与优化参数 ---
    parser.add_argument('--angle_dim', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--angle_weight', type=float, default=1.0)
    parser.add_argument('--seq_weight', type=float, default=1.0)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=25)
    parser.add_argument('--score_threshold', type=int, default=400)

    # --- 消融实验参数 (Flags) ---
    parser.add_argument('--no_esm', action='store_true', help="消融：禁用 ESM 特征")
    parser.add_argument('--ablate_resolution', action='store_true', help="消融：禁用分辨率感知 (Resolution-Aware)")
    parser.add_argument('--no_bidirectional', action='store_true', help="消融：禁用双向数据增强")

    args = parser.parse_args()
    
    # 处理逻辑反转
    args.use_esm = not args.no_esm
    args.bidirectional = not args.no_bidirectional
    
    # 动态占位符 (后面会自动检测)
    args.context_dim = 0 
    
    train(args)