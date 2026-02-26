# setup_high_confidence_testset.py
import os
import pandas as pd
import requests
from tqdm import tqdm
import argparse
import time

def get_fasta_from_uniprot(uniprot_id):
    """
    通过UniProt API在线获取指定ID的FASTA格式序列。
    """
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    try:
        response = requests.get(url, timeout=10) # 设置10秒超时
        response.raise_for_status() # 如果请求失败 (如404)，则抛出异常
        return response.text
    except requests.exceptions.RequestException as e:
        # print(f"警告: 无法获取 {uniprot_id} 的FASTA序列. 原因: {e}")
        return None

def setup_testset_and_fasta(links_file, num_samples, score_threshold, output_dir):
    """
    一个完整的函数，完成筛选、抽样、下载、建文件夹和保存FASTA的全部流程。
    """
    # --- 1. 读取和筛选数据 ---
    print(f"步骤 1/4: 正在从 '{links_file}' 读取并筛选数据...")
    df = pd.read_csv(links_file, sep='\t')
    df_high_confidence = df[df['combined_score'] >= score_threshold]
    
    if len(df_high_confidence) == 0:
        print(f"错误: 找不到任何 combined_score >= {score_threshold} 的数据。请检查阈值或文件。")
        return
        
    print(f"找到 {len(df_high_confidence)} 个分数高于 {score_threshold} 的相互作用对。")

    # --- 2. 随机抽样 ---
    if len(df_high_confidence) < num_samples:
        print(f"警告: 高置信度样本数({len(df_high_confidence)})少于请求的样本数({num_samples})。将使用所有高置信度样本。")
        test_df = df_high_confidence
    else:
        test_df = df_high_confidence.sample(n=num_samples, random_state=42) # 使用固定随机种子保证可重复性
    
    testset_path = os.path.join(output_dir, 'test_set_pairs.tsv')
    test_df.to_csv(testset_path, sep='\t', index=False)
    print(f"步骤 2/4: 成功创建测试集，包含 {len(test_df)} 个相互作用对，已保存到 '{testset_path}'。")

    # --- 3. 提取唯一ID并下载FASTA ---
    p1_ids = set(test_df['protein1_uniprot_id'].unique())
    p2_ids = set(test_df['protein2_uniprot_id'].unique())
    all_uniprot_ids = sorted(list(p1_ids.union(p2_ids)))
    
    print(f"步骤 3/4: 共 {len(all_uniprot_ids)} 个唯一蛋白质ID需要下载FASTA序列。")
    fasta_data_dir = os.path.join(output_dir, 'fasta_by_id')
    os.makedirs(fasta_data_dir, exist_ok=True)
    
    all_fasta_content = ""
    for uniprot_id in tqdm(all_uniprot_ids, desc="下载并保存FASTA"):
        fasta_content = get_fasta_from_uniprot(uniprot_id)
        if fasta_content:
            # a. 保存到单独的子目录文件中
            protein_dir = os.path.join(fasta_data_dir, uniprot_id)
            os.makedirs(protein_dir, exist_ok=True)
            with open(os.path.join(protein_dir, f"{uniprot_id}.fasta"), 'w') as f:
                f.write(fasta_content)
            
            # b. 追加到总的FASTA内容中
            all_fasta_content += fasta_content
        
        # 为了不给UniProt服务器太大压力，每次请求后稍微暂停一下
        time.sleep(0.1) 
        
    # --- 4. 保存总的FASTA文件 ---
    total_fasta_path = os.path.join(output_dir, 'inputs_for_sota.fasta')
    with open(total_fasta_path, 'w') as f:
        f.write(all_fasta_content)
        
    print(f"步骤 4/4: 所有序列已合并保存到 '{total_fasta_path}'。")
    print("\n数据准备完成！")
    print(f"下一步，您可以使用 '{total_fasta_path}' 作为SOTA模型的输入，")
    print(f"并使用 '{testset_path}' 作为您自己模型的测试集输入。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="自动创建高质量测试集并从UniProt下载FASTA序列。")
    
    parser.add_argument('--links_file', type=str, 
                        default="/home/featurize/work/protein/human_protein_interactions_verified_200.tsv",
                        help="包含所有相互作用对的TSV文件路径。")
    
    parser.add_argument('--num_samples', type=int, default=200, 
                        help="要从高置信度数据中随机抽取的样本数量。")
                        
    parser.add_argument('--score_threshold', type=int, default=400, 
                        help="用于筛选的最低combined_score。")

    parser.add_argument('--output_dir', type=str, default="./high_confidence_testset",
                        help="存放所有输出文件（测试集TSV、FASTA文件等）的目录。")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    setup_testset_and_fasta(args.links_file, args.num_samples, args.score_threshold, args.output_dir)