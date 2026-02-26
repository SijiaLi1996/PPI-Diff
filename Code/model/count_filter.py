import os

def count_folders_with_atom(directory):
    folder_count = 0
    try:
        # 遍历 output 目录下的每个子文件夹
        for folder_name in os.listdir(directory):
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                # 检查每个文件夹中的文件
                for root, dirs, files in os.walk(folder_path):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        with open(file_path, 'r') as file:
                            for line in file:
                                if line.startswith("ATOM"):
                                    folder_count += 1
                                    break  # 找到 ATOM 后停止读取该文件夹
                            else:
                                continue
                            break
    except Exception as e:
        print(f"An error occurred: {e}")
    return folder_count

# 使用示例
base_directory = '/home/featurize/work/protein/drug_protein'  # 替换为你的 output 文件夹路径
result = count_folders_with_atom(base_directory)
print(f"Number of folders containing files with ATOM records: {result}")