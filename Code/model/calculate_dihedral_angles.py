import os
import numpy as np
from Bio.PDB import PDBParser, PPBuilder
# 导入 pdb_filter 模块
import pdb_filter

# 1. 读取PDB文件 (移动到 pdb_filter.py)
# def read_pdb_file(pdb_file):
#     ...

# 2. 计算六面角信息 (与之前的代码相同)
def calculate_dihedral_angles(coords):
    """
    计算蛋白质的二面角（六面角）信息
    """
    num_atoms = len(coords)
    dihedral_angles = []
    for i in range(3, num_atoms):
        p0 = coords[i - 3]
        p1 = coords[i - 2]
        p2 = coords[i - 1]
        p3 = coords[i]

        b1 = p1 - p0
        b2 = p2 - p1
        b3 = p3 - p2

        # Normalize vectors
        b1 /= np.linalg.norm(b1)
        b2 /= np.linalg.norm(b2)
        b3 /= np.linalg.norm(b3)

        # Compute normals to the planes
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)

        # Calculate the dihedral angle
        m = np.cross(n1, b2)
        x = np.dot(n1, n2)
        y = np.dot(m, n2)
        dihedral = np.degrees(np.arctan2(y, x))

        dihedral_angles.append(dihedral)

    return np.array(dihedral_angles)

# 3. 计算键长信息 (可选) (与之前的代码相同)
def calculate_bond_lengths(coords):
    """
    计算蛋白质的键长信息
    """
    num_atoms = len(coords)
    bond_lengths = []
    for i in range(1, num_atoms):
        p1 = coords[i - 1]
        p2 = coords[i]
        bond_length = np.linalg.norm(p2 - p1)
        bond_lengths.append(bond_length)
    return np.array(bond_lengths)

# 4. 提取序列位置信息 (与之前的代码相同)
def get_sequence_position_embeddings(sequence):
    """
    获取序列位置编码
    """
    seq_length = len(sequence)
    position_embeddings = np.arange(seq_length) / seq_length  # 简单的线性缩放
    return position_embeddings

# 5. 提取 UniProt ID (移动到 pdb_filter.py)
# def get_uniprot_id(pdb_file):
#     ...

# 6. 数据筛选 (移动到 pdb_filter.py)
# def filter_pdb_files(pdb_files, sequence_length_threshold=50, resolution_threshold=4.0, max_structures_per_uniprot=10):
#     ...

# 7. 主程序 (修改)
if __name__ == "__main__":
    # 设置PDB文件目录
    pdb_dir = "path/to/your/pdb/files"  # 替换为你的PDB文件目录

    # 获取所有PDB文件
   # 获取所有PDB文件
    pdb_files = [os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.startswith("filtered_") and f.endswith(".pdb")]

    # 数据筛选
    filtered_pdb_files = pdb_filter.filter_pdb_files(pdb_files, max_structures_per_uniprot=5)  # 设置每个 UniProt ID 最多保留 5 个结构

    # 处理筛选后的PDB文件
    for pdb_file in filtered_pdb_files:
        try:
            # 读取PDB文件
            coords, sequence = pdb_filter.read_pdb_file(pdb_file)

            # 计算六面角信息
            dihedral_angles = calculate_dihedral_angles(coords)

            # 计算键长信息 (可选)
            bond_lengths = calculate_bond_lengths(coords)

            # 获取序列位置信息
            position_embeddings = get_sequence_position_embeddings(sequence)

            # TODO: 将提取的信息保存到文件或数据结构中
            # 可以使用 numpy.save, pandas.DataFrame 等方法保存数据
            output_file = pdb_file.replace(".pdb", ".npz")  # 例如，保存为 .npz 文件

            np.savez(
                output_file,
                coords=coords,
                sequence=sequence,
                dihedral_angles=dihedral_angles,
                bond_lengths=bond_lengths,
                position_embeddings=position_embeddings,
            )

            print(f"Processed and saved data for {pdb_file} to {output_file}")

        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")