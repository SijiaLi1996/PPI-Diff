import numpy as np

# 替换为你的 .npz 文件路径
file_path = "/home/featurize/work/protein/protein_features_by_uniprot/P24855/4AWN_A_features.npz"

try:
    data = np.load(file_path)
    print(f"成功加载 {file_path}")
    print(f"文件中的键: {data.files}")  # 打印 .npz 文件中的键
    if 'angle_features' in data:
        print(f"angle_features 的形状: {data['angle_features'].shape}")
    else:
        print("警告：文件中缺少 'angle_features' 键！")
except FileNotFoundError:
    print(f"错误：文件未找到 {file_path}")
except Exception as e:
    print(f"加载文件时发生错误: {e}")