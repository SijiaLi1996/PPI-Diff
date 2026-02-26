import os
import csv
import re

def normalize_res_num(res_num):
    """Normalize residue number to integer, handling potential letter suffixes."""
    if re.match(r"^\d+$", res_num):  # 纯数字
        try:
            return int(res_num)
        except ValueError:
            return None
    elif re.match(r"^\d+[a-zA-Z]$", res_num):  # 数字 + 字母
        try:
            return int(res_num[:-1])
        except ValueError:
            return None
    else:
        return None

def load_id_mapping(csv_file):
    """Load DrugBank ID to UniProt ID mapping from CSV file."""
    id_mapping = {}
    try:
        with open(csv_file, 'r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                drugbank_id = row['DrugBank ID'].strip()
                uniprot_id = row['UniProt ID'].strip()
                id_mapping[drugbank_id] = uniprot_id
    except FileNotFoundError:
        print(f"ID 映射文件未找到: {csv_file}")
        return None
    except KeyError:
        print(f"CSV 文件缺少必要的列 (DrugBank ID 或 UniProt ID): {csv_file}")
        return None
    except Exception as e:
        print(f"加载 ID 映射文件时发生错误: {e}")
        return None
    return id_mapping

def extract_data(pdb_file, output_file, target_uniprot_id):
    """Extract relevant ATOM lines based on DBREF information."""
    dbref_data = []
    atom_lines = []
    resolution_line = None
    filtered_dbref_lines = []  # 存储匹配的 DBREF 行

    try:
        # 读取 DBREF 信息
        with open(pdb_file, 'r') as infile:
            for line in infile:
                if line.startswith("DBREF"):
                    parts = line.split()

                    if len(parts) >= 8:
                        chain_id = parts[2]
                        res_num_start = parts[3]
                        res_num_end = parts[4]
                        current_uniprot_id = parts[6]

                        # 检查DBREF中的UniProt ID是否与目标UniProt ID匹配
                        if current_uniprot_id == target_uniprot_id:
                            start_res = normalize_res_num(res_num_start)
                            end_res = normalize_res_num(res_num_end)
                            dbref_data.append((chain_id, start_res, end_res))
                            filtered_dbref_lines.append(line)  # 存储 DBREF 行

                elif line.startswith("REMARK   2 RESOLUTION.") and resolution_line is None:
                    resolution_line = line

        # 检查是否找到任何 DBREF 数据
        if not dbref_data:
            print(f"未找到 UniProt ID {target_uniprot_id} 的 DBREF 信息: {pdb_file}")
            return

        # 提取 ATOM 行
        with open(pdb_file, 'r') as infile:
            for line in infile:
                if line.startswith("ATOM"):
                    atom_chain_id = line[21].strip()
                    res_num = normalize_res_num(line[22:26].strip())

                    if res_num is None:
                        continue

                    for chain_id, start_res, end_res in dbref_data:
                        if atom_chain_id == chain_id and start_res <= res_num <= end_res:
                            atom_lines.append(line)
                            break

        # 如果有 ATOM 数据，写入 RESOLUTION、DBREF 和 ATOM 行
        if atom_lines:
            with open(output_file, 'w') as outfile:
                if resolution_line:
                    outfile.write(resolution_line)
                outfile.writelines(filtered_dbref_lines)  # 写入 DBREF 行
                outfile.writelines(atom_lines)
            print(f"成功提取数据并保存到: {output_file}")
        else:
            print(f"未找到与 UniProt ID {target_uniprot_id} 匹配的 ATOM 行: {pdb_file}")

    except FileNotFoundError:
        print(f"文件未找到: {pdb_file}")
    except IOError as e:
        print(f"IO 错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

def process_all_pdbs(directory, id_mapping_file):
    """Process all PDB files in the given directory."""
    id_mapping = load_id_mapping(id_mapping_file)
    if id_mapping is None:
        return

    try:
        for folder_name in os.listdir(directory):
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.pdb'):
                        input_file = os.path.join(folder_path, file_name)
                        output_file = os.path.join(folder_path, f'filtered_{file_name.lstrip("filtered_")}')

                        drugbank_id = folder_name  # 使用 DrugBank ID
                        if drugbank_id in id_mapping:
                            target_uniprot_id = id_mapping[drugbank_id]
                            extract_data(input_file, output_file, target_uniprot_id)
                        else:
                            print(f"DrugBank ID {drugbank_id} 未在 ID 映射文件中找到。")
    except KeyboardInterrupt:
        print("\n用户中断了进程。")
    except Exception as e:
        print(f"发生错误: {e}")

# 使用示例
base_directory = r'/home/featurize/work/protein/drug_protein'
id_mapping_file = r'/home/featurize/data/unique_uniprot_links.csv'
process_all_pdbs(base_directory, id_mapping_file)