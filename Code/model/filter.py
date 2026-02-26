import os

def normalize_res_num(res_num):
    """Normalize residue number to integer, handling potential letter suffixes."""
    if not res_num: # 添加对空字符串的检查
        return None
    try:
        # 尝试直接转换为整数
        return int(res_num)
    except ValueError:
        # 如果直接转换失败，检查最后一个字符是否是字母
        if len(res_num) > 1 and not res_num[-1].isdigit():
            try:
                # 尝试去掉最后一个字母后转换
                return int(res_num[:-1])
            except ValueError:
                return None # 如果去掉字母后仍然转换失败
        else:
            return None # 如果只有一个字符且不是数字，或者格式不符合预期

def extract_data(pdb_file, output_file, uniprot_id="O14920"):
    """Extract relevant ATOM lines based on DBREF information."""
    dbref_data = []

    # 读取 DBREF 信息
    with open(pdb_file, 'r') as infile:
        for line in infile:
            if line.startswith("DBREF"):
                parts = line.split()
                if len(parts) >= 8:
                    chain_id = parts[2]
                    start_res = normalize_res_num(parts[3])
                    end_res = normalize_res_num(parts[4])
                    current_uniprot_id = parts[6]
                    if current_uniprot_id == uniprot_id:
                        dbref_data.append((chain_id, start_res, end_res))

    # 检查是否找到任何 DBREF 数据
    if not dbref_data:
        return

    # 提取 ATOM 行并缓存 RESOLUTION 行
    with open(pdb_file, 'r') as infile:
        atom_lines = []
        resolution_line = None
        for line in infile:
            if line.startswith("REMARK   2 RESOLUTION.") and resolution_line is None:
                resolution_line = line

            elif line.startswith("DBREF"):
                parts = line.split()
                if len(parts) >= 8:
                    chain_id = parts[2]
                    current_uniprot_id = parts[6]
                    if current_uniprot_id == uniprot_id:
                        atom_lines.append(line)

            elif line.startswith("ATOM"):
                # 添加长度检查，确保行长度足够
                if len(line) > 26:  # ATOM行应该至少有27个字符
                    atom_chain_id = line[21].strip()
                    res_num = normalize_res_num(line[22:26].strip())

                    if res_num is None:
                        continue

                    for chain_id, start_res, end_res in dbref_data:
                        if atom_chain_id == chain_id and start_res <= res_num <= end_res:
                            atom_lines.append(line)
                            break

    # 如果有 ATOM 数据，写入 RESOLUTION 和 ATOM 行
    if atom_lines:
        with open(output_file, 'w') as outfile:
            if resolution_line:
                outfile.write(resolution_line)
            outfile.writelines(atom_lines)

def process_all_pdbs(directory):
    """Process all PDB files in the given directory."""
    try:
        for folder_name in os.listdir(directory):
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.pdb'):
                        try:
                            input_file = os.path.join(folder_path, file_name)
                            output_file = os.path.join(folder_path, f'filtered_{file_name.lstrip("filtered_")}')

                            uniprot_id = folder_name
                            extract_data(input_file, output_file, uniprot_id)
                        except Exception as e:
                            print(f"处理文件 {input_file} 时出错: {e}")
    except KeyboardInterrupt:
        print("\n用户中断了进程。")
    except Exception as e:
        print(f"发生错误: {e}")

# 使用示例
base_directory = r'/home/featurize/work/protein/ppi_pdb_by_uniprot'
process_all_pdbs(base_directory)