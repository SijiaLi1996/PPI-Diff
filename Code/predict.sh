#!/bin/bash


echo " 开始运行 PPI-Diff 推理流程..."

# 运行推理脚本，指定示例数据和模型权重的路径
python model/inference.py \
    --links_file dataset/example_links.tsv \
    --features_dir Example/features \
    --pdb_dir Example/pdbs \
    --checkpoint checkpoints/checkpoint_epoch_500.pth \
    --output_dir final_results \
    --max_samples 10

echo " 预测完成！生成的多肽 PDB 文件和 CSV 结果已保存在 final_results 目录下。"