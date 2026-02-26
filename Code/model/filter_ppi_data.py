import pandas as pd
import os

def filter_protein_interactions_hardcoded():
    """
    读取蛋白质相互作用TSV文件，筛选出两个蛋白质长度都小于指定最大长度的相互作用对。
    输入和输出文件路径以及最大长度都硬编码在函数内部。
    """
    # 硬编码输入文件路径
    input_tsv_file = "/home/featurize/work/protein/human_protein_interactions_length_filtered.tsv"
    # 硬编码输出文件路径
    output_tsv_file = "/home/featurize/work/protein/human_protein_interactions_filtered_100.tsv"
    # 硬编码最大长度
    max_length = 100

    if not os.path.exists(input_tsv_file):
        print(f"Error: Input file not found at {input_tsv_file}")
        return

    print(f"Loading data from {input_tsv_file}...")
    df = pd.read_csv(input_tsv_file, sep='\t')
    print(f"Original data shape: {df.shape}")

    df['protein1_length'] = pd.to_numeric(df['protein1_length'], errors='coerce')
    df['protein2_length'] = pd.to_numeric(df['protein2_length'], errors='coerce')

    df.dropna(subset=['protein1_length', 'protein2_length'], inplace=True)

    filtered_df = df[
        (df['protein1_length'] < max_length) &
        (df['protein2_length'] < max_length)
    ]

    print(f"Filtered data shape (both proteins < {max_length}): {filtered_df.shape}")

    output_dir = os.path.dirname(output_tsv_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    filtered_df.to_csv(output_tsv_file, sep='\t', index=False)
    print(f"Filtered data saved to {output_tsv_file}")

if __name__ == "__main__":
    filter_protein_interactions_hardcoded()