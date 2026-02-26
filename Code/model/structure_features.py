import os
import numpy as np
import math
import warnings
# import shutil # 不再需要，因为不复制文件了

from Bio.PDB import PDBParser, PPBuilder, Polypeptide
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import seq1 # 用于三字母码转单字母码
import protein_geometry # 导入你的角度计算模块

# --- 处理可选的 pdb_filter 模块导入 ---
try:
    # 尝试从名为 pdb_filter.py 的文件导入 filter_pdb_files 函数
    from pdb_filter import filter_pdb_files, get_uniprot_id  # 导入 get_uniprot_id
    print("信息：成功从 'pdb_filter.py' 导入 'filter_pdb_files' 和 'get_uniprot_id' 函数。")
    PDB_FILTER_AVAILABLE = True # 标记筛选函数可用
except ImportError:
    print("警告：无法从 'pdb_filter.py' 导入 'filter_pdb_files' 和 'get_uniprot_id' 函数。筛选功能将不可用。")
    PDB_FILTER_AVAILABLE = False # 标记筛选函数不可用
    # 定义一个虚拟函数，以便后续代码在调用时不会崩溃
    def filter_pdb_files(pdb_files, **kwargs):
        print("信息：虚拟 filter_pdb_files 被调用，返回所有文件。")
        return pdb_files # 虚拟函数直接返回原始列表
    def get_uniprot_id(pdb_file):  # 定义一个虚拟的 get_uniprot_id
        print("信息：虚拟 get_uniprot_id 被调用，返回 None。")
        return None

# 忽略 Bio.PDB 解析 PDB 文件时可能产生的一些警告
warnings.simplefilter('ignore', PDBConstructionWarning)

# --- 特征提取辅助函数 ---

# one_hot_encode_sequence 函数的定义已被移除，因为它不再被使用
# def one_hot_encode_sequence(sequence):
#     """对氨基酸序列进行 One-Hot 编码。"""
#     # ... (此函数已不再需要)

def get_relative_position_embeddings(seq_length):
    """获取相对序列位置编码 (线性归一化到 0 到 1)。"""
    if seq_length == 0:
        return np.array([]).reshape(0, 1)
    # 如果长度为 1，返回 [0.0]；否则，从 0 到 1 线性分布
    position_embeddings = np.arange(seq_length) / (seq_length - 1) if seq_length > 1 else np.array([0.0])
    return position_embeddings.reshape(-1, 1) # 返回列向量

def extract_dbref_mapping(header, chain_id):
    """
    从 PDB 文件头部的 'dbrefs' 字典中解析指定链的 DBREF 映射信息 (直接文本解析)。
    这个函数现在只返回 header 和 chain_id，传递给 get_uniprot_id
    """
    return header, chain_id # 只返回 header 和 chain_id

# --- 核心特征提取函数 ---
def get_structure_features(pdb_file):
    """
    从 PDB 文件读取结构，提取并整合多种特征，包括六个主链角度。
    只处理第一个模型中找到的第一个多肽链。
    只包含有 CA 原子的标准氨基酸残基。
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        model = structure[0]
        header = structure.header

        ppb = PPBuilder()
        peptides = ppb.build_peptides(model)

        if not peptides:
            # print(f"信息：在 {pdb_file} 中未找到多肽链。")
            return None

        poly = peptides[0] # 处理第一条链
        if not poly: # 检查多肽链是否为空
             # print(f"信息：在 {pdb_file} 中找到的多肽链为空。")
             return None
        # 通过第一个残基获取链ID，并检查残基是否存在
        try:
            chain_id = poly[0].get_parent().id
        except IndexError:
            # print(f"信息：在 {pdb_file} 中找到的多肽链不包含残基。")
            return None # 如果链没有残基，也无法处理

        # --- 计算角度 ---
        raw_angles_list = protein_geometry.calculate_backbone_angles(poly)

        # --- 收集有效残基信息 ---
        valid_residues_data = []
        original_residue_indices = [] # 存储原始多肽链中的索引
        for i, residue in enumerate(poly):
            res_name = residue.get_resname()
            # 检查是否是标准氨基酸并且存在 C-alpha 原子
            if Polypeptide.is_aa(res_name, standard=True) and "CA" in residue:
                try:
                    # 使用 Bio.SeqUtils.seq1 进行转换
                    aa = seq1(res_name)
                except KeyError:
                    # 如果 seq1 无法识别 res_name，打印警告并跳过该残基
                    print(f"警告：无法将残基名 '{res_name}' (链 {chain_id}, PDB ID {residue.id[1]}) 转换为单字母代码。跳过此残基。")
                    continue # 跳到下一个残基

                ca_coord = residue["CA"].get_coord()
                pdb_res_id = residue.id[1] # PDB 文件中的残基编号

                # 存储数据
                valid_residues_data.append({
                    "aa": aa,
                    "coord": ca_coord,
                    "pdb_id": pdb_res_id,
                })
                original_residue_indices.append(i) # 存储原始索引 'i'

        if not valid_residues_data:
            # print(f"信息：在 {pdb_file} 的链 {chain_id} 中未找到有效的标准氨基酸残基。")
            return None

        # --- 后处理 ---
        sequence = "".join([data["aa"] for data in valid_residues_data])
        coords = np.array([data["coord"] for data in valid_residues_data])
        pdb_ids = [data["pdb_id"] for data in valid_residues_data] # 原始 PDB 残基编号
        seq_length = len(sequence) # 有效残基的数量

        # --- 处理角度 ---
        # 确保 raw_angles_list 和 original_residue_indices 长度匹配或索引有效
        if len(raw_angles_list) != len(poly):
             print(f"警告：原始角度列表长度({len(raw_angles_list)})与多肽链长度({len(poly)})不匹配，可能导致索引错误。")
             # 这里我们选择返回 None，因为角度数据可能不准确
             return None

        try:
            # 仅提取有效残基对应的角度
            filtered_raw_angles = [raw_angles_list[i] for i in original_residue_indices]
        except IndexError:
             print(f"错误：提取有效残基对应角度时发生索引错误 (链 {chain_id})。原始角度列表长度 {len(raw_angles_list)}，最大索引 {max(original_residue_indices) if original_residue_indices else -1}。")
             return None # 索引错误，无法继续

        angle_features = protein_geometry.process_angles_sin_cos(filtered_raw_angles) # Shape (n, 12)

        # --- 其他特征 ---
        # 不再计算 sequence_encoded
        # sequence_encoded = one_hot_encode_sequence(sequence) # 这一行被移除

        relative_positions = get_relative_position_embeddings(seq_length) # (n, 1)

        # --- 绝对位置 ---
        # absolute_positions 应该就是 pdb_ids，即 PDB 文件中原始的残基编号
        absolute_positions = np.array(pdb_ids).reshape(-1, 1) # (n, 1)

        # --- 获取 UniProt ID 和分辨率 ---
        uniprot_id = None
        resolution = np.nan # 初始化分辨率为 NaN

        if PDB_FILTER_AVAILABLE: # 只有当 pdb_filter 可用时才尝试获取
            uniprot_id = get_uniprot_id(pdb_file) # 使用 pdb_filter.py 中的函数

            # 尝试从 PDB header 中获取分辨率
            # Bio.PDB 的 header 字典中通常有 'resolution' 键
            if 'resolution' in header and isinstance(header['resolution'], (float, int)):
                resolution = float(header['resolution'])
            else:
                # 如果 header 中没有，或者格式不对，则保持为 NaN
                print(f"警告：未能从 PDB header 中获取分辨率信息 for {pdb_file}。")
                resolution = np.nan

        # --- 验证长度 ---
        # 不再包含 sequence_encoded 的长度检查
        if not (seq_length == coords.shape[0] ==
                angle_features.shape[0] == relative_positions.shape[0] ==
                absolute_positions.shape[0]):
             raise ValueError(f"内部错误：链 {chain_id} 的特征长度不一致！"
                             f"Seq: {seq_length}, Coords: {coords.shape[0]}, "
                             f"Angles: {angle_features.shape[0]}, "
                             f"RelPos: {relative_positions.shape[0]}, AbsPos: {absolute_positions.shape[0]}")

        # --- 融合特征 ---
        # fused_features 现在只包含结构相关的特征（不含One-Hot序列）
        fused_features = np.concatenate(
            [angle_features,         # (n, 12)
             relative_positions,     # (n, 1)
             absolute_positions      # (n, 1)
            ],
            axis=1
        )
        feature_dim = fused_features.shape[1]
        print(f"结构特征已融合 (不含One-Hot序列)，形状: ({seq_length}, {feature_dim})")

        # --- 返回结果 ---
        # 不再返回 sequence_encoded
        return (sequence, coords, angle_features, fused_features,
                relative_positions, absolute_positions, uniprot_id, chain_id, resolution) # 添加 resolution

    except FileNotFoundError:
        print(f"错误：PDB 文件未找到 {pdb_file}")
        return None
    except Exception as e:
        import traceback
        print(f"处理 {pdb_file} 时发生意外错误: {e}")
        traceback.print_exc()
        return None


# --- 主程序 (main block) ---
if __name__ == "__main__":
    # --- 配置参数 ---
    root_search_dir = "/home/featurize/work/protein/ppi_pdb_by_uniprot" # 搜索 PDB 文件的根目录
    base_output_dir = "./protein_features_by_ppi200_" # 保存特征文件的基础目录 <<<--- 请确认或修改此路径
    pdb_extensions = (".pdb",)
    filename_prefix = "filtered_" # 用于查找输入文件
    apply_pdb_filter = True # 是否启用筛选
    sequence_length_threshold = 0 # 筛选参数
    resolution_threshold = 4.0   # 筛选参数
    max_structures_per_uniprot = 15 # 筛选参数

    # --- 程序开始 ---
    print(f"开始递归搜索 PDB 文件，根目录: {root_search_dir}")
    print(f"查找前缀为 '{filename_prefix}' 且扩展名为 {pdb_extensions} 的文件。")
    print(f"特征将保存到基础目录: {base_output_dir} (按下 UniProt ID 分子目录)")

    # 确保基础输出目录存在
    if not os.path.exists(base_output_dir):
        try:
            os.makedirs(base_output_dir)
            print(f"已创建基础输出目录: {base_output_dir}")
        except OSError as e:
            print(f"错误：无法创建基础输出目录 '{base_output_dir}': {e}")
            exit() # 无法创建输出目录，退出

    # 1. 使用 os.walk() 递归获取所有符合条件的 PDB 文件路径
    all_pdb_files = []
    if not os.path.isdir(root_search_dir):
         print(f"错误：根目录 '{root_search_dir}' 不是一个有效的目录。")
         exit()
    for dirpath, dirnames, filenames in os.walk(root_search_dir):
        for filename in filenames:
            if filename.startswith(filename_prefix) and filename.lower().endswith(pdb_extensions):
                full_path = os.path.join(dirpath, filename)
                all_pdb_files.append(full_path)

    if not all_pdb_files:
        print(f"警告：在目录 {root_search_dir} 及其子目录下未找到任何符合条件的文件。")
        # 可以选择在这里退出，或者让程序继续（但不会处理任何文件）
        # exit()
    else:
        print(f"找到 {len(all_pdb_files)} 个符合条件的 PDB 文件。")


    # 2. (可选) 数据筛选
    if apply_pdb_filter and PDB_FILTER_AVAILABLE:
        print(f"应用筛选：序列长度 >= {sequence_length_threshold}, 分辨率 <= {resolution_threshold}, 每个 UniProt ID 最多 {max_structures_per_uniprot} 个结构。")
        try:
            # 调用从 pdb_filter.py 导入的函数
            filtered_pdb_files = filter_pdb_files(
                all_pdb_files,
                sequence_length_threshold=sequence_length_threshold,
                resolution_threshold=resolution_threshold,
                max_structures_per_uniprot=max_structures_per_uniprot
            )
            print(f"筛选后剩余 {len(filtered_pdb_files)} 个 PDB 文件进行处理。")
        except Exception as filter_e:
            # 如果筛选函数本身执行出错
            print(f"错误：调用筛选函数 filter_pdb_files 时出错: {filter_e}")
            print(f"将跳过筛选，处理所有 {len(all_pdb_files)} 个找到的文件...")
            # !!! 关键：出错时也要赋值 !!!
            filtered_pdb_files = all_pdb_files
    else:
        # 如果不满足筛选条件 (apply_pdb_filter=False 或 函数不可用)
        if apply_pdb_filter and not PDB_FILTER_AVAILABLE:
             # 如果想筛选但函数不可用
             print("警告：筛选已启用但 'filter_pdb_files' 函数无法从 'pdb_filter.py' 导入。将处理所有文件。")
        else: # apply_pdb_filter is False
             print("信息：未启用 PDB 筛选 (apply_pdb_filter=False)。将处理所有文件。")
        # !!! 关键：不筛选或无法筛选时也要赋值 !!!
        filtered_pdb_files = all_pdb_files

    # 3. 循环处理筛选后的 PDB 文件
    processed_count = 0
    error_count = 0
    if not filtered_pdb_files:
         print("\n没有文件需要处理。")

    for pdb_file in filtered_pdb_files:
        base_filename = os.path.basename(pdb_file) # 例如 "filtered_1E4J.pdb"
        print(f"\n--- 处理文件: {pdb_file} ---")

        # 调用核心函数获取特征
        result = get_structure_features(pdb_file)

        if result is None:
            print(f"未能从 {base_filename} 提取有效特征，已跳过。")
            error_count += 1fuse
            continue

        # 解包结果，获取 uniprot_id 和 resolution
        # 注意：这里不再解包 sequence_encoded
        (sequence, coords, angle_features, fused_features,
         relative_positions, absolute_positions, uniprot_id, chain_id, resolution) = result

        seq_length = len(sequence)
        print(f"成功提取链 {chain_id} 的特征，有效残基数: {seq_length}")
        if uniprot_id:
            print(f"关联的 UniProt ID: {uniprot_id}")
        else:
            print(f"警告：未能确定 UniProt ID。")
        if not np.isnan(resolution): # 检查是否是 NaN
            print(f"PDB 分辨率: {resolution:.2f} Å")
        else:
            print(f"警告：未能确定 PDB 分辨率。")


        # --- 确定 UniProt ID 文件夹 ---
        if uniprot_id and isinstance(uniprot_id, str) and uniprot_id.strip():
            uniprot_folder_name = uniprot_id.strip()
        else:
            uniprot_folder_name = "Unknown_UniProt"
        uniprot_output_subdir = os.path.join(base_output_dir, uniprot_folder_name)
        os.makedirs(uniprot_output_subdir, exist_ok=True) # 确保目录存在

        # 5. 保存结果到 .npz 文件 (在 UniProt ID 子目录中，文件名无前缀)
        pdb_id_part_with_prefix = base_filename.split('.')[0] # e.g., "filtered_1E4J"
        if pdb_id_part_with_prefix.startswith(filename_prefix):
            pdb_id_part_no_prefix = pdb_id_part_with_prefix[len(filename_prefix):] # e.g., "1E4J"
        else:
            pdb_id_part_no_prefix = pdb_id_part_with_prefix # 如果没有前缀，则保留原样

        # 构建最终的 .npz 文件名 (不带前缀)
        output_filename = f"{pdb_id_part_no_prefix}_{chain_id}_features.npz" # e.g., "1E4J_B_features.npz"
        output_path = os.path.join(uniprot_output_subdir, output_filename) # 路径包含 UniProt ID 子目录

        np.savez_compressed(
            output_path, # 使用新的路径
            fused_features=fused_features, # 现在不含 One-Hot 序列
            sequence=sequence, # 原始序列字符串，供 ESM 使用
            uniprot_id=str(uniprot_id) if uniprot_id else "Unknown",
            chain_id=chain_id,
            pdb_filename=base_filename, # 仍然保存原始带前缀的文件名
            pdb_filepath=pdb_file, # 原始完整路径
            pdb_id_processed=pdb_id_part_no_prefix, # 保存处理后的PDB ID (无前缀)
            absolute_positions=absolute_positions.flatten(),
            coords=coords,
            angle_features=angle_features,  # 添加 angle_features
            relative_positions=relative_positions.flatten(), # 添加 relative_positions
            resolution=resolution # 添加分辨率
        )
        print(f"特征已保存到: {output_path}")
        processed_count += 1 # 只有成功保存特征才计数

    # --- 结束 ---
    print("\n===================================")
    print("处理完成。")
    print(f"成功保存特征文件的数量: {processed_count}")
    print(f"处理过程中跳过或出错的总文件数: {error_count}")
    print(f"总计尝试处理文件数: {len(filtered_pdb_files)}") # 基于筛选后的列表长度
    print("===================================")