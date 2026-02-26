import pandas as pd
import requests
import os
import logging
import shutil
import time
import random
import csv
import re

# 配置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_res_num(res_num):
    """Normalize residue number to integer, handling potential letter suffixes."""
    if isinstance(res_num, (int, float)):
        return int(res_num)
    if isinstance(res_num, str):
        res_num = res_num.strip()
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
    return None

def extract_pdb_ids_from_uniprot(uniprot_id):
    """
    从 UniProt REST API 获取指定 UniProt ID 的数据，并从中提取 PDB IDs 和分辨率。

    Args:
        uniprot_id (str): UniProt ID (例如 "P01050").

    Returns:
        list: 包含 PDB ID 和分辨率的元组的列表，如果找到，否则返回 None。
             例如: [('1ABC', '2.0'), ('1XYZ', '2.5')]
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        uniprot_text = response.text
        logging.debug(f"Successfully retrieved UniProt data for {uniprot_id}")

        pdb_data = []
        for line in uniprot_text.splitlines():
            if line.startswith("DR   PDB;"):
                parts = line.split(";")
                if len(parts) >= 4:
                    pdb_id = parts[1].strip()
                    resolution_str = parts[3].strip().replace(' A', '')
                    try:
                        resolution = float(resolution_str)
                    except ValueError:
                        resolution = None
                    pdb_data.append((pdb_id, resolution))
                    logging.debug(f"Extracted PDB ID: {pdb_id}, Resolution: {resolution} from UniProt data")
        if pdb_data:
            return pdb_data
        else:
            logging.info(f"No PDB IDs found in UniProt data for {uniprot_id}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching UniProt data for {uniprot_id}: {e}")
        return None

def download_pdb_file(pdb_id, output_dir):
    """
    从RCSB PDB下载指定的PDB文件。

    Args:
        pdb_id (str): PDB ID (例如 "1ABC").
        output_dir (str): 保存PDB文件的目录。

    Returns:
        str: 下载的PDB文件路径，如果成功；否则返回 None。
    """
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")

    if os.path.exists(output_path):
        logging.info(f"PDB file {pdb_id} already exists at {output_path}, skipping download.")
        return output_path

    try:
        response = requests.get(pdb_url)
        response.raise_for_status()

        with open(output_path, "w") as f:
            f.write(response.text)
        logging.info(f"Downloaded PDB file {pdb_id} to {output_path}")
        return output_path

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading PDB file {pdb_id}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error writing PDB file {pdb_id}: {e}")
        return None

def extract_and_filter_pdb_data(pdb_file, output_file, target_uniprot_id):
    """
    Extract relevant ATOM lines based on DBREF information and filter.

    Args:
        pdb_file (str): Input PDB file path.
        output_file (str): Output filtered PDB file path.
        target_uniprot_id (str): Target UniProt ID for filtering.

    Returns:
        bool: True if filtering was successful and output file was created, False otherwise.
    """
    dbref_data = []
    atom_lines = []
    resolution_line = None
    filtered_dbref_lines = []

    # 如果没有提供 target_uniprot_id，则无法进行 DBREF 过滤
    if not target_uniprot_id:
        logging.warning(f"No target UniProt ID provided for filtering {pdb_file}. Skipping DBREF filtering.")
        # 你可以在这里决定如何处理：例如，直接复制文件
        # try:
        #     shutil.copy2(pdb_file, output_file)
        #     logging.info(f"Copied original PDB file to {output_file} as no target UniProt ID for filtering.")
        #     return True
        # except Exception as e:
        #      logging.error(f"Error copying file: {e}")
        return False


    try:
        with open(pdb_file, 'r') as infile:
            for line in infile:
                if line.startswith("DBREF"):
                    parts = line.split()
                    if len(parts) >= 8:
                        chain_id = parts[2].strip()
                        res_num_start = parts[3].strip()
                        res_num_end = parts[4].strip()
                        current_uniprot_id = parts[6].strip()

                        if current_uniprot_id == target_uniprot_id:
                            start_res = normalize_res_num(res_num_start)
                            end_res = normalize_res_num(res_num_end)
                            if start_res is not None and end_res is not None:
                                dbref_data.append((chain_id, start_res, end_res))
                                filtered_dbref_lines.append(line)
                                logging.debug(f"Matched DBREF: Chain ID={chain_id}, Res Start={start_res}, Res End={end_res}, UniProt ID={current_uniprot_id}")


                elif line.startswith("REMARK   2 RESOLUTION.") and resolution_line is None:
                    resolution_line = line.strip() + "\n" # Keep the newline

        if not dbref_data:
            logging.warning(f"No DBREF information found for UniProt ID {target_uniprot_id} in {pdb_file}")
            return False

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

        if atom_lines:
            with open(output_file, 'w') as outfile:
                if resolution_line:
                    outfile.write(resolution_line)
                outfile.writelines(filtered_dbref_lines)
                outfile.writelines(atom_lines)
            logging.info(f"Successfully extracted and filtered data to: {output_file}")
            return True
        else:
            logging.warning(f"No ATOM lines found matching DBREF information for UniProt ID {target_uniprot_id} in {pdb_file}")
            return False

    except FileNotFoundError:
        logging.error(f"File not found during filtering: {pdb_file}")
        return False
    except IOError as e:
        logging.error(f"IO error during filtering: {e}")
        return False
    except Exception as e:
        logging.error(f"An unknown error occurred during filtering: {e}")
        return False

def is_likely_pdb_id(id_string):
    """
    Heuristic to check if a string is likely a PDB ID.
    PDB IDs are 4 characters, typically alphanumeric.
    """
    if isinstance(id_string, str) and len(id_string) == 4:
        return id_string.isalnum()
    return False


def process_drug_protein_mapping(id_mapping_file, output_base_dir):
    """
    Processes DrugBank to UniProt/PDB ID mapping, downloads and filters PDBs.
    Prioritizes UniProt ID processing, falls back to PDB ID if UniProt processing fails.
    """
    mapping_data = []
    try:
        with open(id_mapping_file, 'r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                drugbank_id = row['DrugBank ID'].strip()
                uniprot_or_pdb_id = row['UniProt ID'].strip()
                # 检查 DrugBank ID 是否非空
                if drugbank_id:
                    mapping_data.append((drugbank_id, uniprot_or_pdb_id))
                else:
                    logging.warning(f"Skipping row with empty DrugBank ID.")

    except FileNotFoundError:
        logging.error(f"ID mapping file not found: {id_mapping_file}")
        return
    except KeyError:
        logging.error(f"CSV file missing required columns (DrugBank ID or UniProt ID): {id_mapping_file}")
        return
    except Exception as e:
        logging.error(f"Error loading ID mapping file: {e}")
        return

    if not mapping_data:
        logging.warning("No valid DrugBank to UniProt/PDB mappings found in the CSV file.")
        return

    # 创建主输出目录
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    for drugbank_id, uniprot_or_pdb_id in mapping_data:
        logging.info(f"Processing DrugBank ID: {drugbank_id}, ID: {uniprot_or_pdb_id}")

        drug_dir = os.path.join(output_base_dir, drugbank_id)

        # 只有当 uniprot_or_pdb_id 非空时才进行处理
        if not uniprot_or_pdb_id:
            logging.warning(f"Skipping processing for DrugBank ID {drugbank_id} due to empty UniProt ID/PDB ID.")
            continue

        if not os.path.exists(drug_dir):
            os.makedirs(drug_dir)

        successfully_processed = False

        # --- 优先按 UniProt ID 处理 ---
        logging.info(f"Attempting to process {uniprot_or_pdb_id} as a UniProt ID for {drugbank_id}")
        target_uniprot_id_for_filter = uniprot_or_pdb_id

        # 检查是否已经存在过滤后的文件 (针对这个 UniProt ID 关联的所有 PDB)
        associated_pdb_data = extract_pdb_ids_from_uniprot(uniprot_or_pdb_id)
        if associated_pdb_data:
            # Check if any of the associated PDBs already have a filtered file
            if any(os.path.exists(os.path.join(drug_dir, f'filtered_{pdb_id}.pdb')) for pdb_id, _ in associated_pdb_data):
                 logging.info(f"Filtered PDB file already exists for {drugbank_id} (based on UniProt ID {uniprot_or_pdb_id}), skipping.")
                 successfully_processed = True # Treat as successfully processed if filtered file exists
            else:
                logging.debug(f"Found PDB data: {associated_pdb_data} for {drugbank_id}")

                for pdb_id, resolution in associated_pdb_data:
                    logging.info(f"Attempting to process PDB ID: {pdb_id} for {drugbank_id} (from UniProt)")

                    # 下载 PDB 文件
                    pdb_file_path = download_pdb_file(pdb_id, drug_dir)

                    if pdb_file_path:
                        # 构建过滤后的输出文件路径
                        filtered_output_file = os.path.join(drug_dir, f'filtered_{pdb_id}.pdb')

                        # 进行过滤操作，使用从 CSV 获取的 UniProt ID
                        if extract_and_filter_pdb_data(pdb_file_path, filtered_output_file, target_uniprot_id_for_filter):
                            logging.info(f"Successfully processed and filtered PDB ID: {pdb_id} for {drugbank_id} (from UniProt)")
                            successfully_processed = True
                            # 如果成功处理，可以选择删除原始下载的 PDB 文件以节省空间
                            # os.remove(pdb_file_path)
                            break # 成功处理一个 PDB 后，停止尝试其他 PDB

                    time.sleep(random.uniform(2, 4)) # 在每次尝试下载和过滤之间添加延迟
        else:
             logging.info(f"No PDB IDs found for {drugbank_id} (UniProt ID: {uniprot_or_pdb_id}) via UniProt API.")


        # --- 如果 UniProt ID 处理失败，尝试按 PDB ID 处理 ---
        if not successfully_processed:
            logging.info(f"UniProt ID processing failed for {drugbank_id}. Attempting to process {uniprot_or_pdb_id} as a PDB ID.")

            if is_likely_pdb_id(uniprot_or_pdb_id):
                logging.info(f"ID {uniprot_or_pdb_id} for {drugbank_id} is likely a PDB ID.")
                target_pdb_id = uniprot_or_pdb_id
                target_uniprot_id_for_filter = None # We don't have a target UniProt ID from the CSV

                # 检查是否已经存在过滤后的文件 (针对这个直接的 PDB ID)
                filtered_output_file = os.path.join(drug_dir, f'filtered_{target_pdb_id}.pdb')
                if os.path.exists(filtered_output_file):
                    logging.info(f"Filtered PDB file {filtered_output_file} already exists, skipping.")
                    successfully_processed = True # Treat as successfully processed if filtered file exists
                else:
                    # 直接尝试下载和过滤这个 PDB ID
                    pdb_file_path = download_pdb_file(target_pdb_id, drug_dir)

                    if pdb_file_path:
                         # 注意：在这里进行过滤时，我们需要知道要过滤的 UniProt ID。
                         # 如果 CSV 中直接给的是 PDB ID，我们可能没有对应的 UniProt ID 来进行 DBREF 过滤。
                         # 这种情况下，extract_and_filter_pdb_data 将无法进行 DBREF 过滤。
                         logging.warning(f"Cannot perform DBREF filtering for PDB ID {target_pdb_id} as no target UniProt ID is provided in the CSV.")
                         # 如果你的目标是直接保存这个 PDB 文件，可以在这里复制文件
                         # try:
                         #     shutil.copy2(pdb_file_path, filtered_output_file)
                         #     logging.info(f"Copied original PDB file to {filtered_output_file} as no target UniProt ID for filtering.")
                         #     successfully_processed = True
                         # except Exception as e:
                         #      logging.error(f"Error copying file: {e}")
                         pass # 如果不复制也不过滤，就什么也不做，让 successfully_processed 保持 False

                    time.sleep(random.uniform(2, 4))
            else:
                logging.warning(f"ID {uniprot_or_pdb_id} for {drugbank_id} is neither a recognized UniProt ID nor a likely PDB ID.")


        if not successfully_processed:
            logging.warning(f"Could not successfully process any relevant PDB file for {drugbank_id} (ID: {uniprot_or_pdb_id})")


        time.sleep(random.uniform(2, 4)) # 在处理每个 DrugBank ID 之间添加延迟


# 使用示例
base_directory = r'/home/featurize/work/protein/drug_protein'
id_mapping_file = r'/home/featurize/data/unique_uniprot_links.csv'
process_drug_protein_mapping(id_mapping_file, base_directory)