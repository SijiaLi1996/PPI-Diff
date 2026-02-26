import os
from Bio.PDB import PDBParser, PPBuilder
from Bio import SwissProt  # 用于解析 UniProt ID

# 对 PDB 文件进行筛选，选择符合特定标准的 PDB 文件，用于后续的分析和建模
def read_pdb_file(pdb_file):
    """
    读取PDB文件，返回原子坐标和序列信息
    """
    parser = PDBParser(QUIET=True)  # 设置 QUIET=True 避免输出冗余信息
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]
    # 提取原子坐标
    coords = []
    for atom in model.get_atoms():
        coords.append(atom.get_coord())
    # 提取序列信息
    ppb = PPBuilder()
    sequences = []
    for pp in ppb.build_peptides(model):
        sequences.append(pp.get_sequence())
    sequence = "".join(map(str,sequences)) # 将 Seq 对象转换为字符串

    return coords, sequence

def get_uniprot_id(pdb_file):
    """
    从 PDB 文件中提取 UniProt ID (模仿提供的代码)。
    """
    try:
        with open(pdb_file, 'r') as infile:
            for line in infile:
                if line.startswith("DBREF"):
                    parts = line.split()
                    if len(parts) >= 8:
                        uniprot_id = parts[6]  # 尝试使用 parts[6] 提取 UniProt ID
                        # print(f"  [调试] 解析到的 UniProt ID: {uniprot_id}, 完整 DBREF 行: {line.strip()}")
                        return uniprot_id
        return None  # 如果没有找到 UniProt ID，返回 None
    except FileNotFoundError:
        print(f"Error: File not found: {pdb_file}")
        return None
    except Exception as e:
        print(f"Error extracting UniProt ID from {pdb_file}: {e}")
        return None
    
# In pdb_filter.py

def get_resolution(pdb_file):
    """
    从PDB文件中提取分辨率信息，适配 "REMARK 2 RESOLUTION. VALUE ANGSTROMS." 格式。
    """
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                # 检查行是否以标准前缀开头
                if line.startswith("REMARK   2 RESOLUTION."):
                    # 按空格分割行内容
                    parts = line.split()
                    # parts 示例: ['REMARK', '2', 'RESOLUTION.', '3.20', 'ANGSTROMS.']
                    # 或者 ['REMARK', '2', 'RESOLUTION.', 'NOT', 'APPLICABLE']

                    # 检查分割后的列表长度是否足够，并且 RESOLUTION. 后面的部分看起来像数字
                    # 通常分辨率值在索引 3 的位置
                    if len(parts) > 3:
                        resolution_text = parts[3]
                        try:
                            # 尝试将该部分转换为浮点数
                            resolution = float(resolution_text)
                            return resolution
                        except ValueError:
                            # 如果转换失败 (例如遇到 "NOT", "NULL"),
                            # 说明这一行没有有效的数字分辨率，继续查找下一行
                            # print(f"信息：在 {pdb_file} 找到非数字分辨率文本: '{resolution_text}'") # 可选调试
                            continue
                    # else: # 如果分割后部分不够，说明格式不符，继续查找
                        # print(f"信息：在 {pdb_file} 找到格式不符的 REMARK 2 行: {line.strip()}") # 可选调试
                        # continue # 省略，循环会自动继续

            # 遍历完文件未找到有效分辨率行或有效数值
            return None
    except FileNotFoundError:
        print(f"Error: File not found when extracting resolution: {pdb_file}")
        return None
    except Exception as e:
        # 捕获其他可能的读取或处理错误
        print(f"Error reading or processing resolution from {pdb_file}: {e}")
        return None

# --- 其他 pdb_filter.py 中的函数 (read_pdb_file, get_uniprot_id, filter_pdb_files) 保持不变 ---
# ...
def filter_pdb_files(pdb_files, sequence_length_threshold=0, resolution_threshold=4.0, max_structures_per_uniprot=10):
    """
    根据序列长度、分辨率和 UniProt ID 筛选 PDB 文件
    """
    filtered_pdb_files = []
    uniprot_structures = {}  # 用于存储每个 UniProt ID 对应的 PDB 文件和分辨率

    for pdb_file in pdb_files:
        try:
            coords, sequence = read_pdb_file(pdb_file)
            
            # 获取当前PDB文件的序列长度
            current_sequence_length = len(sequence)
            #---------------------------------------
            resolution = get_resolution(pdb_file)

            print(f"处理文件: {pdb_file}")
            print(f"  序列长度: {current_sequence_length}, 分辨率: {resolution}")
           #--------------------------------------------------------------------
            # 根据序列长度筛选
            #只有当sequence_length_threshold小于等于实际序列长度时才进行筛选
            if sequence_length_threshold > current_sequence_length :
                print(f"阈值: sequence_length_threshold = {sequence_length_threshold}")  # 关键调试行
                print(f"  排除原因: 序列长度 {current_sequence_length} 小于阈值 {sequence_length_threshold}")
                continue
                
            # 提取 UniProt ID
            uniprot_id = get_uniprot_id(pdb_file) # 确保在分辨率检查之前提取 UniProt ID

            # 打印 UniProt ID，用于调试
            print(f"  提取的 UniProt ID: {uniprot_id}")

            if uniprot_id is None:
                print(f"  排除原因: 缺少 UniProt ID")
                continue  # 如果没有 UniProt ID，则跳过该文件
                
            # 从PDB文件中提取分辨率信息            
            if resolution is None:
                print(f"  排除原因: 缺少分辨率信息")
                continue  # 如果没有分辨率信息，则跳过该文件
                
            if resolution > resolution_threshold:
                print(f"  排除原因: 分辨率 {resolution} 大于阈值 {resolution_threshold}")
                continue


            # 存储 UniProt ID 和 PDB 文件
            if uniprot_id not in uniprot_structures:
                uniprot_structures[uniprot_id] = []
            uniprot_structures[uniprot_id].append((pdb_file, resolution))

        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")

    # 对每个 UniProt ID，选择分辨率最高的 N 个 PDB 文件
    for uniprot_id, structures in uniprot_structures.items():
        # 按分辨率排序 (升序)
        structures.sort(key=lambda x: x[1])
        # 选择分辨率最高的 N 个 PDB 文件
        selected_structures = structures[:max_structures_per_uniprot]
        # 将选择的 PDB 文件添加到 filtered_pdb_files 列表中 
        filtered_pdb_files.extend([pdb_file for pdb_file, _ in selected_structures])

    return filtered_pdb_files