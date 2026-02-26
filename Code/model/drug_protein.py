import pandas as pd
import requests
import os
import logging
import shutil
from bs4 import BeautifulSoup
import time
import random
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import webbrowser

# 配置日志记录
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
                pdb_id = parts[1].strip()
                resolution = parts[3].strip().replace(' A', '')  # 提取分辨率并移除 " A"
                try:
                    resolution = float(resolution)  # 转换为浮点数
                except ValueError:
                    resolution = None  # 如果无法转换为浮点数，则设置为 None
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

def get_pdb_sequence_length(pdb_id, uniprot_id):
    """
    从 RCSB PDB API 获取 PDB 文件的序列长度。

    Args:
        pdb_id (str): PDB ID (例如 "1ABC").
        uniprot_id (str): UniProt ID (例如 "P01050").

    Returns:
        int: 序列长度，如果成功获取；否则返回 None。
    """
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/1"  # 使用 RCSB 提供的 API
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # 尝试从 aligned_regions 获取序列长度
        if 'rcsb_polymer_entity_group_membership' in data:
            aligned_regions = data['rcsb_polymer_entity_group_membership']
            for region in aligned_regions:
                if region['aggregation_method'] == 'matching_uniprot_accession' and region['group_id'] == uniprot_id:
                    length = region['aligned_regions'][0]['length']
                    logging.debug(f"Successfully retrieved sequence length for PDB ID: {pdb_id}, length: {length} from aligned_regions")
                    return length

        logging.warning(f"Could not determine sequence length for PDB ID: {pdb_id} from aligned_regions.")

        return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching sequence data for PDB file {pdb_id}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing sequence data for PDB file {pdb_id}: {e}")
        return None

def download_pdb_file(pdb_id, output_dir):
    """
    从RCSB PDB下载指定的PDB文件。

    Args:
        pdb_id (str): PDB ID (例如 "1ABC").
        output_dir (str): 保存PDB文件的目录。
    """
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")

    try:
        response = requests.get(pdb_url)
        response.raise_for_status()

        with open(output_path, "w") as f:
            f.write(response.text)
        logging.info(f"Downloaded PDB file {pdb_id} to {output_path}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading PDB file {pdb_id}: {e}")
    except Exception as e:
        logging.error(f"Error writing PDB file {pdb_id}: {e}")


def main():
    """
    主函数：从CSV文件读取DrugBank ID 和 UniProt ID 的对应关系，
    然后下载与这些UniProt ID关联的PDB文件中长度最长的，如果长度相同要分辨率最高的数据，并保存到以 DrugBank ID 命名的文件夹中。
    """
    output_dir = "drug_protein"
    csv_file_path = "/home/featurize/data/unique_uniprot_links.csv"

    # 创建主输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 检查CSV文件是否存在
    if not os.path.exists(csv_file_path):
        logging.error(f"CSV file not found at: {csv_file_path}")
        return

    try:
        # 读取CSV文件
        df = pd.read_csv(csv_file_path)
        logging.info(f"Successfully loaded data from {csv_file_path}")

        # 确保所需的列存在
        if 'DrugBank ID' not in df.columns or 'UniProt ID' not in df.columns:
            logging.error("CSV file must contain 'DrugBank ID' and 'UniProt ID' columns.")
            return

        # 遍历CSV文件中的每一行
        for index, row in df.iterrows():
            drugbank_id = row['DrugBank ID']
            uniprot_id = row['UniProt ID']

            if pd.isna(drugbank_id) or pd.isna(uniprot_id):
                logging.warning(f"Skipping row {index} due to missing DrugBank ID or UniProt ID.")
                continue

            logging.debug(f"Processing DrugBank ID: {drugbank_id}, UniProt ID: {uniprot_id}")

            # 从 UniProt REST API 获取 PDB IDs
            pdb_data = extract_pdb_ids_from_uniprot(uniprot_id)

            if pdb_data:
                logging.debug(f"Found PDB data: {pdb_data} for {drugbank_id}")

                # 创建以DrugBank ID命名的子目录
                drug_dir = os.path.join(output_dir, drugbank_id)
                if not os.path.exists(drug_dir):
                    os.makedirs(drug_dir)

                # 找到最佳 PDB ID
                best_pdb_id = None
                max_length = 0
                best_resolution = float('inf')  # 初始最佳分辨率设置为无穷大

                for pdb_id, resolution in pdb_data:
                    logging.debug(f"Processing PDB ID: {pdb_id}, Resolution: {resolution} for {drugbank_id}")
                    length = get_pdb_sequence_length(pdb_id, uniprot_id)
                    logging.debug(f"Sequence length for PDB ID {pdb_id}: {length}")
                    if length is not None:
                        if length > max_length:
                            max_length = length
                            best_pdb_id = pdb_id
                            best_resolution = resolution if resolution is not None else float('inf')
                            logging.debug(f"New best PDB ID: {best_pdb_id}, Length: {max_length}, Resolution: {best_resolution}")
                        elif length == max_length and resolution is not None and resolution < best_resolution:
                            best_pdb_id = pdb_id
                            best_resolution = resolution
                            logging.debug(f"New best PDB ID (same length, better resolution): {best_pdb_id}, Length: {max_length}, Resolution: {best_resolution}")

                if best_pdb_id:
                    download_pdb_file(best_pdb_id, drug_dir)
                    logging.info(f"Downloaded best PDB file {best_pdb_id} for {drugbank_id}")
                else:
                    logging.info(f"Could not determine best PDB file for {drugbank_id}")
            else:
                logging.info(f"No PDB IDs found for {drugbank_id} (UniProt ID: {uniprot_id})")

            time.sleep(random.uniform(2, 4))  # 添加随机延迟

    except FileNotFoundError:
        logging.error(f"Error: The file {csv_file_path} was not found.")
    except pd.errors.EmptyDataError:
        logging.error(f"Error: The file {csv_file_path} is empty.")
    except pd.errors.ParserError:
        logging.error(f"Error: Could not parse the file {csv_file_path}. Please check the file format.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()