import os
import numpy as np
import math
import warnings
import csv # 导入 csv 模块
import requests # 导入 requests 库用于下载
import time # 用于下载时的延迟

from Bio.PDB import PDBParser, PPBuilder, Polypeptide
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import seq1 # 用于三字母码转单字母码
import protein_geometry # 导入你的角度计算模块

# --- 处理可选的 pdb_filter 模块导入 ---
# 在这个新的流程中，pdb_filter 的作用可能需要调整，
# 因为我们是先下载再处理。暂时保留，但其筛选逻辑可能需要适配下载后的文件列表。
try:
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

# --- 特征提取辅助函数 (保持不变) ---

def one_hot_encode_sequence(sequence):
    """对氨基酸序列进行 One-Hot 编码。"""
    amino_acids = sorted(Polypeptide.protein_letters_3to1.values())
    aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
    num_amino_acids = len(amino_acids)

    encoded_sequence = []
    for aa in sequence:
        one_hot = np.zeros(num_amino_acids)
        if aa in aa_to_int:
            one_hot[aa_to_int[aa]] = 1
        encoded_sequence.append(one_hot)
    return np.array(encoded_sequence)

def get_relative_position_embeddings(seq_length):
    """获取相对序列位置编码 (线性归一化到 0 到 1)。"""
    if seq_length == 0:
        return np.array([]).reshape(0, 1)
    position_embeddings = np.arange(seq_length) / (seq_length - 1) if seq_length > 1 else np.array([0.0])
    return position_embeddings.reshape(-1, 1)

# 在新的流程中，我们通过 CSV 文件获取 UniProt ID，所以这个函数不再直接用于获取 UniProt ID
# 但保留其原始功能，如果需要从 PDB 文件头获取其他信息
def extract_dbref_mapping(header, chain_id):
    """
    从 PDB 文件头部的 'dbrefs' 字典中解析指定链的 DBREF 映射信息 (直接文本解析)。
    这个函数现在只返回 header 和 chain_id，传递给 get_uniprot_id
    """
    return header, chain_id # 只返回 header 和 chain_id

# --- PDB 下载和处理函数 ---

def download_pdb_for_uniprot(uniprot_id, drugbank_id, output_dir):
    """
    根据 UniProt ID 查询并下载相关的 PDB 文件到以 DrugBank ID 命名的子目录。
    返回下载的 PDB 文件路径列表。
    """
    print(f"正在为 UniProt ID '{uniprot_id}' (DrugBank ID: {drugbank_id}) 查找并下载 PDB 文件...")
    pdb_files_downloaded = []
    drugbank_output_subdir = os.path.join(output_dir, drugbank_id)
    os.makedirs(drugbank_output_subdir, exist_ok=True)

    # --- 实现 PDB 查询和下载逻辑 ---
    # 这部分需要根据您使用的 PDB 数据库 API 或服务进行修改。
    # 以下是一个使用 RCSB PDB Data API 的示例思路：

    # 示例：查询与 UniProt ID 关联的 PDB ID
    # 这是一个简化的示例，实际查询可能更复杂，需要处理分页、筛选等
    search_url = "https://search.rcsb.org/rcsbsearch/v1/query"
    query = {
        "query": {
            "type": "terminal",
            "service": "text_chem",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequences.database_accession",
                "operator": "exact_match",
                "value": uniprot_id
            }
        },
        "return_type": "entry"
    }

    try:
        response = requests.post(search_url, json=query)
        response.raise_for_status() # 检查 HTTP 错误
        pdb_ids = response.json().get("result_set", [])
        print(f"找到 {len(pdb_ids)} 个与 UniProt ID '{uniprot_id}' 相关的 PDB ID。")

        # 示例：下载每个 PDB 文件
        for entry in pdb_ids:
            pdb_id = entry.get("identifier")
            if not pdb_id:
                continue

            pdb_download_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            output_pdb_path = os.path.join(drugbank_output_subdir, f"{pdb_id}.pdb")

            if os.path.exists(output_pdb_path):
                print(f"文件 '{pdb_id}.pdb' 已存在，跳过下载。")
                pdb_files_downloaded.append(output_pdb_path)
                continue

            print(f"正在下载 PDB ID '{pdb_id}'...")
            try:
                pdb_response = requests.get(pdb_download_url, stream=True)
                pdb_response.raise_for_status()

                with open(output_pdb_path, 'wb') as f:
                    for chunk in pdb_response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"成功下载 '{pdb_id}.pdb'。")
                pdb_files_downloaded.append(output_pdb_path)
                # 增加延迟，避免对服务器造成过大压力
                time.sleep(1)

            except requests.exceptions.RequestException as e:
                print(f"下载 PDB ID '{pdb_id}' 时发生错误: {e}")
                continue # 跳过当前文件，继续下一个

    except requests.exceptions.RequestException as e:
        print(f"查询 RCSB PDB API 时发生错误: {e}")
    except Exception as e:
        print(f"处理 UniProt ID '{uniprot_id}' 的下载过程时发生意外错误: {e}")

    return pdb_files_downloaded

# --- 核心特征提取函数 (修改以适应新的流程) ---
# 在这个函数中，我们将直接解析 PDB 文件，并尝试从 DBREF 获取 UniProt ID，
# 但主要的 UniProt ID 来源是 CSV 文件。我们将把 CSV 中的 UniProt ID 作为优先。

def get_structure_features(pdb_file, expected_uniprot_id=None):
    """
    从 PDB 文件读取结构，提取并整合多种特征，包括六个主链角度。
    只处理第一个模型中找到的第一个多肽链。
    只包含有 CA 原子的标准氨基酸残基。
    可以传入预期的 UniProt ID (从 CSV 获取)。
    """
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        model = structure[0]
        header = structure.header

        ppb = PPBuilder()
        peptides = ppb.build_peptides(model)

        if not peptides:
            return None

        poly = peptides[0]
        if not poly:
             return None

        try:
            chain_id = poly[0].get_parent().id
        except IndexError:
            return None

        # --- 尝试从 PDB DBREF 获取 UniProt ID (作为补充) ---
        uniprot_id_from_dbref = None
        try:
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith("DBREF"):
                        # 解析 DBREF 行 (根据 PDB 文件格式规范)
                        pdb_id_in_dbref = line[7:11].strip()
                        chain_id_in_dbref = line[12].strip()
                        database_name = line[26:32].strip()
                        accession = line[33:41].strip()

                        if chain_id_in_dbref == chain_id and database_name == "UNP":
                            uniprot_id_from_dbref = accession
                            # print(f"从 DBREF 获取到 UniProt ID: {uniprot_id_from_dbref} (链 {chain_id})")
                            break

        except FileNotFoundError:
            pass # 文件未找到，忽略
        except Exception as e:
            print(f"警告：解析 {pdb_file} 的 DBREF 时发生错误: {e}")

        # 使用传入的 expected_uniprot_id 作为主要的 UniProt ID
        final_uniprot_id = expected_uniprot_id if expected_uniprot_id else uniprot_id_from_dbref
        if not final_uniprot_id:
             print(f"警告：未能确定 {os.path.basename(pdb_file)} 的 UniProt ID。")
             # 可以选择在这里返回 None 或者继续处理但 UniProt ID 为 None/Unknown
             # 为了保存文件，我们最好有一个 UniProt ID 或标记
             final_uniprot_id = "Unknown_UniProt"


        # --- 计算角度 ---
        raw_angles_list = protein_geometry.calculate_backbone_angles(poly)

        # --- 收集有效残基信息 ---
        valid_residues_data = []
        original_residue_indices = []
        for i, residue in enumerate(poly):
            res_name = residue.get_resname()
            if Polypeptide.is_aa(res_name, standard=True) and "CA" in residue:
                try:
                    aa = seq1(res_name)
                except KeyError:
                    print(f"警告：无法将残基名 '{res_name}' (链 {chain_id}, PDB ID {residue.id[1]}) 转换为单字母代码。跳过此残基。")
                    continue

                ca_coord = residue["CA"].get_coord()
                pdb_res_id = residue.id[1]

                valid_residues_data.append({
                    "aa": aa,
                    "coord": ca_coord,
                    "pdb_id": pdb_res_id,
                })
                original_residue_indices.append(i)

        if not valid_residues_data:
            return None

        # --- 后处理 ---
        sequence = "".join([data["aa"] for data in valid_residues_data])
        coords = np.array([data["coord"] for data in valid_residues_data])
        pdb_ids = [data["pdb_id"] for data in valid_residues_data]
        seq_length = len(sequence)

        # --- 处理角度 ---
        if len(raw_angles_list) != len(poly):
             print(f"警告：原始角度列表长度({len(raw_angles_list)})与多肽链长度({len(poly)})不匹配，可能导致索引错误。")
             pass

        try:
            filtered_raw_angles = [raw_angles_list[i] for i in original_residue_indices]
        except IndexError:
             print(f"错误：提取有效残基对应角度时发生索引错误 (链 {chain_id})。原始角度列表长度 {len(raw_angles_list)}，最大索引 {max(original_residue_indices) if original_residue_indices else -1}。")
             return None

        angle_features = protein_geometry.process_angles_sin_cos(filtered_raw_angles)

        # --- 其他特征 ---
        sequence_encoded = one_hot_encode_sequence(sequence)
        relative_positions = get_relative_position_embeddings(seq_length)

        # --- 绝对位置 (仍然简化) ---
        absolute_positions = np.zeros((seq_length, 1)) # 简化为全零

        # --- 验证长度 ---
        if not (seq_length == coords.shape[0] == sequence_encoded.shape[0] ==
                angle_features.shape[0] == relative_positions.shape[0] ==
                absolute_positions.shape[0]):
             raise ValueError(f"内部错误：链 {chain_id} 的特征长度不一致！"
                             f"Seq: {seq_length}, Coords: {coords.shape[0]}, "
                             f"OneHot: {sequence_encoded.shape[0]}, Angles: {angle_features.shape[0]}, "
                             f"RelPos: {relative_positions.shape[0]}, AbsPos: {absolute_positions.shape[0]}")

        # --- 融合特征 ---
        fused_features = np.concatenate(
            [sequence_encoded, angle_features, relative_positions, absolute_positions],
            axis=1
        )
        feature_dim = fused_features.shape[1]
        print(f"特征已融合，形状: ({seq_length}, {feature_dim})")

        # --- 返回结果 ---
        return (sequence, sequence_encoded, coords, angle_features, fused_features,
                relative_positions, absolute_positions, final_uniprot_id, chain_id)

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
    csv_file_path = "/home/featurize/data/uniprot links.csv" # CSV 文件路径
    base_output_dir = "./protein_features_by_drugbank" # 保存特征文件的基础目录 (按 DrugBank ID 分子目录)
    downloaded_pdb_dir = "./downloaded_pdb_files" # 下载的 PDB 文件临时存放目录
    pdb_extensions = (".pdb",)
    # filename_prefix = "filtered_" # 在新的流程中不再需要这个前缀来查找文件
    apply_pdb_filter = True # 是否启用筛选 (将在下载后对文件列表进行筛选)
    sequence_length_threshold = 0 # 筛选参数
    resolution_threshold = 4.0   # 筛选参数
    max_structures_per_uniprot = 10 # 筛选参数

    # --- 程序开始 ---
    print(f"开始从 CSV 文件 '{csv_file_path}' 读取数据。")
    print(f"PDB 文件将下载到临时目录: {downloaded_pdb_dir}")
    print(f"特征将保存到基础目录: {base_output_dir} (按下 DrugBank ID 分子目录)")

    # 确保基础输出目录和下载目录存在
    for directory in [base_output_dir, downloaded_pdb_dir]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"已创建目录: {directory}")
            except OSError as e:
                print(f"错误：无法创建目录 '{directory}': {e}")
                exit()

    # 1. 从 CSV 文件读取 UniProt ID 和 DrugBank ID 映射
    uniprot_drugbank_map = {}
    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            if 'UniProt ID' not in reader.fieldnames or 'DrugBank ID' not in reader.fieldnames:
                print(f"错误：CSV 文件 '{csv_file_path}' 必须包含 'UniProt ID' 和 'DrugBank ID' 列。")
                exit()

            for row in reader:
                uniprot_id = row.get('UniProt ID', '').strip()
                drugbank_id = row.get('DrugBank ID', '').strip()

                if uniprot_id: # 只处理 UniProt ID 不为空的行
                    if drugbank_id:
                        # 存储 UniProt ID 到 DrugBank ID 的映射
                        # 一个 UniProt ID 可能对应多个 DrugBank ID，这里简单存储第一个找到的
                        if uniprot_id not in uniprot_drugbank_map:
                            uniprot_drugbank_map[uniprot_id] = drugbank_id
                        # 如果一个 UniProt ID 对应多个 DrugBank ID，这里只记录第一个，
                        # 如果需要处理所有映射，需要更复杂的逻辑
                    else:
                        print(f"警告：UniProt ID '{uniprot_id}' 没有对应的 DrugBank ID，将跳过。")

    except FileNotFoundError:
        print(f"错误：CSV 文件未找到 '{csv_file_path}'。")
        exit()
    except Exception as e:
        print(f"读取 CSV 文件时发生错误: {e}")
        exit()

    if not uniprot_drugbank_map:
        print("警告：在 CSV 文件中未找到有效的 UniProt ID 和 DrugBank ID 映射。程序将退出。")
        exit()

    print(f"从 CSV 文件中读取到 {len(uniprot_drugbank_map)} 个有效的 UniProt ID 映射。")

    # 2. 根据 UniProt ID 下载 PDB 文件
    all_downloaded_pdb_files = []
    for uniprot_id, drugbank_id in uniprot_drugbank_map.items():
        print(f"\n--- 处理 UniProt ID: {uniprot_id} (DrugBank ID: {drugbank_id}) ---")
        downloaded_files = download_pdb_for_uniprot(uniprot_id, drugbank_id, downloaded_pdb_dir)
        all_downloaded_pdb_files.extend(downloaded_files)

    if not all_downloaded_pdb_files:
        print("\n警告：未能下载任何 PDB 文件。请检查网络连接、RCSB PDB API 状态或 UniProt ID 是否有效。")
        # 可以选择在这里退出，或者让程序继续（但不会处理任何文件）
        # exit()
    else:
        print(f"\n总共下载了 {len(all_downloaded_pdb_files)} 个 PDB 文件。")


    # 3. (可选) 数据筛选 (对下载的文件进行筛选)
    files_to_process = all_downloaded_pdb_files
    if apply_pdb_filter and PDB_FILTER_AVAILABLE:
        print(f"\n应用筛选：序列长度 >= {sequence_length_threshold}, 分辨率 <= {resolution_threshold}, 每个 UniProt ID 最多 {max_structures_per_uniprot} 个结构。")
        try:
            # 调用从 pdb_filter.py 导入的函数，对下载的文件列表进行筛选
            # 注意：filter_pdb_files 可能需要能够处理 UniProt ID 的获取，
            # 或者您需要修改 filter_pdb_files 以接受 UniProt ID 映射
            # 这里假设 filter_pdb_files 能够从 PDB 文件自身获取 UniProt ID 或进行其他判断
            files_to_process = filter_pdb_files(
                all_downloaded_pdb_files,
                sequence_length_threshold=sequence_length_threshold,
                resolution_threshold=resolution_threshold,
                max_structures_per_uniprot=max_structures_per_uniprot
            )
            print(f"筛选后剩余 {len(files_to_process)} 个 PDB 文件进行处理。")
        except Exception as filter_e:
            print(f"错误：调用筛选函数 filter_pdb_files 时出错: {filter_e}")
            print(f"将跳过筛选，处理所有 {len(all_downloaded_pdb_files)} 个下载的文件...")
            files_to_process = all_downloaded_pdb_files
    else:
        if apply_pdb_filter and not PDB_FILTER_AVAILABLE:
             print("警告：筛选已启用但 'filter_pdb_files' 函数无法从 'pdb_filter.py' 导入。将处理所有下载的文件。")
        else:
             print("信息：未启用 PDB 筛选 (apply_pdb_filter=False)。将处理所有下载的文件。")
        files_to_process = all_downloaded_pdb_files


    # 4. 循环处理筛选后的 PDB 文件并提取特征
    processed_count = 0
    error_count = 0
    if not files_to_process:
         print("\n没有文件需要处理。")

    for pdb_file in files_to_process:
        base_filename = os.path.basename(pdb_file) # 例如 "1E4J.pdb"
        print(f"\n--- 处理文件: {pdb_file} ---")

        # 尝试从文件路径确定 DrugBank ID 和 UniProt ID
        # 假设下载的文件保存在 DrugBank ID 目录下
        drugbank_id_from_path = os.path.basename(os.path.dirname(pdb_file))
        # 从 uniprot_drugbank_map 中查找对应的 UniProt ID
        # 需要反向映射或者遍历映射来找到与 drugbank_id_from_path 对应的 UniProt ID
        # 更简单的方法是，在 download_pdb_for_uniprot 中，将 uniprot_id 传递给 get_structure_features
        # 但是 get_structure_features 是处理单个文件的，不知道原始的 uniprot_id
        # 暂时先尝试从 DBREF 获取，或者在 get_structure_features 中使用传入的参数

        # 调用核心函数获取特征，并传入预期的 UniProt ID (这里需要根据 DrugBank ID 找到 UniProt ID)
        # 这是一个挑战，因为一个 DrugBank ID 可能对应多个 UniProt ID
        # 简单的处理方式是，假设每个下载的文件都对应一个 DrugBank ID 和一个 UniProt ID
        # 我们可以尝试从文件名或路径推断，但这依赖于下载时的命名规则
        # 更好的方法是在下载时记录每个下载文件对应的 UniProt ID 和 DrugBank ID

        # 为了简化，我们暂时假设可以通过 DrugBank ID 找到对应的 UniProt ID
        # 实际应用中，您可能需要维护一个更精确的映射
        # 查找与当前 DrugBank ID 关联的 UniProt ID
        current_uniprot_id = None
        for uid, dbid in uniprot_drugbank_map.items():
            if dbid == drugbank_id_from_path:
                current_uniprot_id = uid
                break # 找到第一个匹配的 UniProt ID

        result = get_structure_features(pdb_file, expected_uniprot_id=current_uniprot_id)


        if result is None:
            print(f"未能从 {base_filename} 提取有效特征，已跳过。")
            error_count += 1
            continue

        (sequence, sequence_encoded, coords, angle_features, fused_features,
         relative_positions, absolute_positions, extracted_uniprot_id, chain_id) = result

        seq_length = len(sequence)
        print(f"成功提取链 {chain_id} 的特征，有效残基数: {seq_length}")
        if extracted_uniprot_id:
            print(f"关联的 UniProt ID: {extracted_uniprot_id}")
        else:
            print(f"警告：未能确定 UniProt ID。")

        # --- 确定 DrugBank ID 文件夹 (从路径获取) ---
        drugbank_folder_name = drugbank_id_from_path
        drugbank_output_subdir = os.path.join(base_output_dir, drugbank_folder_name)
        os.makedirs(drugbank_output_subdir, exist_ok=True) # 确保目录存在

        # 5. 保存结果到 .npz 文件 (在 DrugBank ID 子目录中)
        pdb_id_part = base_filename.split('.')[0] # e.g., "1E4J"

        # 构建最终的 .npz 文件名
        output_filename = f"{pdb_id_part}_{chain_id}_features.npz" # e.g., "1E4J_B_features.npz"
        output_path = os.path.join(drugbank_output_subdir, output_filename) # 路径包含 DrugBank ID 子目录

        np.savez_compressed(
            output_path,
            fused_features=fused_features,
            sequence=sequence,
            uniprot_id=str(extracted_uniprot_id) if extracted_uniprot_id else "Unknown",
            chain_id=chain_id,
            pdb_filename=base_filename,
            pdb_filepath=pdb_file,
            pdb_id_processed=pdb_id_part,
            absolute_positions=absolute_positions.flatten(),
            coords=coords,
            angle_features=angle_features,
            relative_positions=relative_positions.flatten()
        )
        print(f"特征已保存到: {output_path}")
        processed_count += 1

    # --- 结束 ---
    print("\n===================================")
    print("处理完成。")
    print(f"成功保存特征文件的数量: {processed_count}")
    print(f"处理过程中跳过或出错的总文件数: {error_count}")
    print(f"总计尝试处理文件数: {len(files_to_process)}") # 基于筛选后的列表长度
    print("===================================")