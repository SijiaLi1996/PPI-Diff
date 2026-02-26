import os
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from Bio import PDB
import logging
import time # 用于下载时的延迟

# 配置日志
logging.basicConfig(level=logging.INFO,  # 设置日志级别为 INFO，以便看到下载信息
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 创建会话对象，设置重试策略
session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1,
    method_whitelist=["HEAD", "GET", "OPTIONS"]
)
session.mount('https://', HTTPAdapter(max_retries=retries))

# 初始化 PDB 解析器 (在下载脚本中可能不需要，但保留以备后续使用)
# pdb_parser = PDB.PDBParser(QUIET=True)
# structure_builder = PDB.StructureBuilder.StructureBuilder()

def get_pdb_from_uniprot(uniprot_id, output_root_dir):
    """
    根据 UniProt ID 从 UniProt 和 RCSB PDB 数据库下载蛋白质结构数据.
    下载的 PDB 文件将保存在以 UniProt ID 命名的子文件夹中。

    Args:
        uniprot_id (str): UniProt ID.
        output_root_dir (str): 保存所有蛋白质结构数据的根目录.
    Returns:
        list: 下载成功的 PDB 文件路径列表.
    """
    pdb_list = []
    downloaded_files = []

    # 为当前的 UniProt ID 创建输出目录
    uniprot_output_dir = os.path.join(output_root_dir, uniprot_id)
    os.makedirs(uniprot_output_dir, exist_ok=True)

    try:
        # 从 UniProt 获取 PDB ID 列表
        logging.info(f"正在从 UniProt 获取 UniProt ID '{uniprot_id}' 的 PDB ID 列表...")
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.txt"
        response = session.get(url)
        response.raise_for_status()

        lines = response.text.splitlines()
        for line in lines:
            if line.startswith("DR   PDB;"):
                parts = line.split(";")
                if len(parts) > 1:
                    pdb_id = parts[1].strip()
                    pdb_list.append(pdb_id)

        if not pdb_list:
            logging.info(f"UniProt ID '{uniprot_id}' 未找到关联的 PDB ID。")
            return downloaded_files

        logging.info(f"UniProt ID '{uniprot_id}' 找到 {len(pdb_list)} 个关联的 PDB ID: {', '.join(pdb_list)}")

        # 下载 PDB 文件到 UniProt ID 的子文件夹
        for pdb_id in pdb_list:
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            pdb_file_path = os.path.join(uniprot_output_dir, f"{pdb_id}.pdb")

            if os.path.exists(pdb_file_path):
                logging.info(f"PDB 文件 '{pdb_id}.pdb' 已存在于 '{uniprot_output_dir}'，跳过下载。")
                downloaded_files.append(pdb_file_path)
                continue

            logging.info(f"正在下载 PDB ID '{pdb_id}' 到 '{uniprot_output_dir}'...")
            try:
                pdb_response = session.get(pdb_url, stream=True) # 使用 stream=True 处理大文件
                pdb_response.raise_for_status()

                with open(pdb_file_path, 'wb') as f:
                    for chunk in pdb_response.iter_content(chunk_size=8192):
                        f.write(chunk)

                logging.info(f"成功下载 PDB 文件: {pdb_file_path}")
                downloaded_files.append(pdb_file_path)
                time.sleep(0.5) # 增加短暂延迟

            except requests.exceptions.RequestException as e:
                logging.error(f"下载 PDB 文件失败: {pdb_id} (UniProt ID: {uniprot_id}) - {e}")
            except Exception as e:
                logging.error(f"处理 PDB ID '{pdb_id}' 下载时发生意外错误 (UniProt ID: {uniprot_id}) - {e}")


    except requests.exceptions.RequestException as e:
        logging.error(f"获取 UniProt 信息失败: {uniprot_id} - {e}")
    except Exception as e:
         logging.error(f"处理 UniProt ID '{uniprot_id}' 时发生意外错误: {e}")

    return downloaded_files


def process_drugbank_uniprot_links(unique_csv_file, links_csv_file, output_root_dir):
    """
    处理 CSV 文件，根据 DrugBank ID 下载关联的 UniProt ID 的蛋白质结构数据.
    下载的 PDB 文件将保存在以 UniProt ID 命名的子文件夹中。

    Args:
        unique_csv_file (str): 包含 DrugBank ID 和 UniProt ID 唯一关联的 CSV 文件路径.
        links_csv_file (str): 包含 DrugBank ID 和 UniProt ID 完整关联的 CSV 文件路径.
        output_root_dir (str): 保存所有蛋白质结构数据的根目录.
    """
    # 1. 从 links_csv_file 构建 DrugBank ID 到 UniProt ID 列表的映射
    drugbank_to_uniprots_map = {}
    try:
        logging.info(f"正在从 '{links_csv_file}' 构建 DrugBank ID 到 UniProt ID 列表的映射...")
        df_links = pd.read_csv(links_csv_file)
        if 'UniProt ID' not in df_links.columns or 'DrugBank ID' not in df_links.columns:
            logging.error(f"错误: CSV 文件 '{links_csv_file}' 必须包含 'UniProt ID' 和 'DrugBank ID' 列。")
            return

        # 过滤掉 UniProt ID 为空的行
        df_links_filtered = df_links.dropna(subset=['UniProt ID'])

        # 构建映射
        for index, row in df_links_filtered.iterrows():
            uniprot_id = str(row['UniProt ID']).strip()
            drugbank_id = str(row['DrugBank ID']).strip()

            if drugbank_id:
                 if drugbank_id not in drugbank_to_uniprots_map:
                     drugbank_to_uniprots_map[drugbank_id] = []
                 if uniprot_id not in drugbank_to_uniprots_map[drugbank_id]: # 避免重复的 UniProt ID
                     drugbank_to_uniprots_map[drugbank_id].append(uniprot_id)

        logging.info(f"从 '{links_csv_file}' 构建了 {len(drugbank_to_uniprots_map)} 个 DrugBank ID 的映射。")

    except FileNotFoundError:
        logging.error(f"错误: CSV 文件未找到: {links_csv_file}")
        return
    except Exception as e:
        logging.error(f"读取 CSV 文件 '{links_csv_file}' 失败: {e}")
        return

    # 2. 从 unique_csv_file 获取要处理的 DrugBank ID 列表
    drugbank_ids_to_process = []
    try:
        logging.info(f"正在从 '{unique_csv_file}' 获取要处理的 DrugBank ID 列表...")
        df_unique = pd.read_csv(unique_csv_file)
        if 'DrugBank ID' not in df_unique.columns or 'UniProt ID' not in df_unique.columns:
             logging.error(f"错误: CSV 文件 '{unique_csv_file}' 必须包含 'DrugBank ID' 和 'UniProt ID' 列。")
             return

        # 过滤掉 UniProt ID 为空的行，并获取对应的 DrugBank ID
        df_unique_filtered = df_unique.dropna(subset=['UniProt ID'])
        drugbank_ids_to_process = df_unique_filtered['DrugBank ID'].unique().tolist()

        logging.info(f"从 '{unique_csv_file}' 获取到 {len(drugbank_ids_to_process)} 个要处理的 DrugBank ID。")

    except FileNotFoundError:
        logging.error(f"错误: CSV 文件未找到: {unique_csv_file}")
        return
    except Exception as e:
        logging.error(f"读取 CSV 文件 '{unique_csv_file}' 失败: {e}")
        return

    # 3. 遍历要处理的 DrugBank ID，下载关联的 PDB 文件
    total_downloaded_files = 0
    processed_uniprot_ids = set() # 记录已经处理过的 UniProt ID，避免重复下载

    for drugbank_id in drugbank_ids_to_process:
        logging.info(f"\n--- 处理 DrugBank ID: {drugbank_id} ---")

        # 获取与当前 DrugBank ID 关联的 UniProt ID 列表
        uniprot_ids_for_drugbank = drugbank_to_uniprots_map.get(drugbank_id, [])

        if not uniprot_ids_for_drugbank:
            logging.warning(f"DrugBank ID '{drugbank_id}' 在 '{links_csv_file}' 中未找到关联的 UniProt ID。跳过。")
            continue

        logging.info(f"DrugBank ID '{drugbank_id}' 关联的 UniProt ID 列表: {', '.join(uniprot_ids_for_drugbank)}")

        # 遍历关联的 UniProt ID，下载 PDB 文件
        for uniprot_id in uniprot_ids_for_drugbank:
            if uniprot_id in processed_uniprot_ids:
                logging.info(f"UniProt ID '{uniprot_id}' 已经被处理过，跳过下载。")
                continue

            downloaded_files = get_pdb_from_uniprot(uniprot_id, output_root_dir)
            total_downloaded_files += len(downloaded_files)
            processed_uniprot_ids.add(uniprot_id) # 标记为已处理

    logging.info("\n===================================")
    logging.info("处理完成。")
    logging.info(f"总共尝试处理 {len(drugbank_ids_to_process)} 个 DrugBank ID。")
    logging.info(f"总共下载了 {total_downloaded_files} 个 PDB 文件。")
    logging.info("===================================")


# 示例用法
unique_csv_file = "/home/featurize/data/unique_uniprot_links.csv"
links_csv_file = "/home/featurize/data/uniprot links.csv"
output_root_dir = "output_protein_by_uniprot" # 保存蛋白质结构的根目录，按 UniProt ID 组织

process_drugbank_uniprot_links(unique_csv_file, links_csv_file, output_root_dir)