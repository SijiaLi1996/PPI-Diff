import os
import torch
import random
import numpy as np
from flask import Flask, request, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename

# 导入您的 PPI_Diff 模块
from PPI_Diff.models import ConditionalAngleDiffusion, SequencePredictor, sample
from PPI_Diff.nerf_utils import reconstruct_backbone_from_angles
from PPI_Diff.fuse_features import fuse_target_features, init_esm_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = 'PPI_Diff/checkpoint_epoch_500.pth'

# 核心优化：在服务器启动时全局初始化 ESM 模型，避免用户每次点击预测都重新加载一遍 ESM，大大提升响应速度
print("Initializing ESM Model for the Web Server...")
init_esm_model()
print("ESM Model loaded successfully!")

def save_pdb_file(coords, sequence, filename, chain_id='B'):
    """保存生成的 Binder PDB (从您的代码原封不动移植)"""
    with open(filename, 'w') as f:
        atom_idx = 1
        for i, res_name in enumerate(sequence):
            res_3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU',
                     'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE',
                     'P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL','X':'UNK'}.get(res_name, 'UNK')
            if coords is not None:
                f.write(f"ATOM  {atom_idx:5d}  N   {res_3} {chain_id}{i+1:4d}    {coords[3*i,0]:8.3f}{coords[3*i,1]:8.3f}{coords[3*i,2]:8.3f}  1.00  0.00           N\n")
                atom_idx += 1
                f.write(f"ATOM  {atom_idx:5d}  CA  {res_3} {chain_id}{i+1:4d}    {coords[3*i+1,0]:8.3f}{coords[3*i+1,1]:8.3f}{coords[3*i+1,2]:8.3f}  1.00  0.00           C\n")
                atom_idx += 1
                f.write(f"ATOM  {atom_idx:5d}  C   {res_3} {chain_id}{i+1:4d}    {coords[3*i+2,0]:8.3f}{coords[3*i+2,1]:8.3f}{coords[3*i+2,2]:8.3f}  1.00  0.00           C\n")
                atom_idx += 1

def save_fasta_file(sequence, filename):
    """保存预测的序列为 FASTA 格式供下载"""
    with open(filename, 'w') as f:
        f.write(f">PPI_Diff_Generated_Binder\n")
        f.write(f"{sequence}\n")

def predict_structure_and_sequence(file_path, output_prefix):
    """处理上传的特征并生成多肽"""
    # 1. 动态特征融合 (调用您的 fuse_features)
    try:
        fused_data = fuse_target_features(
            npz_files=[file_path], 
            target_uniprot_id="Web_Target", # 网页版不需要真实的 UID，用占位符即可
            use_esm=True, 
            ablate_resolution=False 
        )
    except Exception as e:
        raise RuntimeError(f"Feature fusion failed: {e}")
        
    if fused_data is None:
        raise ValueError("Feature fusion returned None. Please check the uploaded .npz file.")
        
    ctx_feat = fused_data['final_features'].to(DEVICE).unsqueeze(0)
    ctx_res = fused_data['resolution_score'].to(DEVICE)
    if ctx_res.dim() == 1: ctx_res = ctx_res.unsqueeze(-1)
    ctx_res = ctx_res.unsqueeze(0)
    
    seq_len = ctx_feat.shape[1]
    
    # 2. 加载模型
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    args = ckpt.get('args', None)
    if args is None:
        raise ValueError("Checkpoint does not contain 'args'. Cannot initialize model.")
    
    # 根据保存的参数初始化
    angle_model = ConditionalAngleDiffusion(
        angle_dim=args.angle_dim, context_dim=args.context_dim, hidden_dim=args.hidden_dim,
        timesteps=args.timesteps, num_layers=args.num_layers
    ).to(DEVICE)
    
    seq_model = SequencePredictor(
        angle_dim=args.angle_dim, context_dim=args.context_dim, hidden_dim=args.hidden_dim,
        vocab_size=args.vocab_size, num_layers=args.num_layers
    ).to(DEVICE)

    if 'angle_model_state_dict' in ckpt:
        angle_model.load_state_dict(ckpt['angle_model_state_dict'])
        seq_model.load_state_dict(ckpt['seq_model_state_dict'])
    else:
        angle_model.load_state_dict(ckpt)
        
    angle_model.eval()
    seq_model.eval()

    # 3. 推理生成过程
    with torch.no_grad():
        # 随机生成 10 到 40 长度的 Binder (对应您的生成逻辑)
        gen_len = random.randint(10, 40)
        mask = torch.zeros((1, seq_len), dtype=torch.bool).to(DEVICE)
        
        gen_ang = sample(
            model=angle_model, context=ctx_feat, resolution=ctx_res, 
            sequence_length=gen_len, batch_size=1, timesteps=1000, 
            context_mask=mask
        )
        logits = seq_model(
            x_angles=gen_ang, context=ctx_feat, resolution=ctx_res, memory_key_padding_mask=mask 
        )
        
        aa_vocab = "ARNDCQEGHILKMFPSTWYV"
        probs = torch.softmax(logits / 0.1, dim=-1)
        pred_indices = torch.multinomial(probs.view(-1, args.vocab_size), num_samples=1).view(-1).cpu().numpy()
        pred_seq = "".join([aa_vocab[i] if i < 20 else 'X' for i in pred_indices])
        
        coords_3d = reconstruct_backbone_from_angles(gen_ang.squeeze(0))
        
    # 4. 保存为 PDB 和 FASTA
    pdb_path = f"{output_prefix}.pdb"
    fasta_path = f"{output_prefix}.fasta"
    
    save_pdb_file(coords_3d, pred_seq, pdb_path, chain_id='B')
    save_fasta_file(pred_seq, fasta_path)
    
    return pdb_path, fasta_path

# ================= 网页路由 =================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        f = request.files['file']
        filename = secure_filename(f.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)

        output_prefix = os.path.splitext(file_path)[0]
        pdb_file, fasta_file = predict_structure_and_sequence(file_path, output_prefix)

        return jsonify({
            "resultMessage": "De novo binder generated successfully!",
            "pdb_download": os.path.basename(pdb_file),
            "fasta_download": os.path.basename(fasta_file)
        })
    except Exception as e:
        return jsonify({"resultMessage": f"Error: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)