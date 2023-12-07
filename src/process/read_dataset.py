import json
import os

from src.utils.debug_message import Assert


# 单个json文件读取
def read_file_json(file_path):
    Assert(os.path.exists(file_path), "Error: JSON file path doesn't exist")

    with open(file_path, 'r') as file:
        data = json.load(file)

    return data

# 批量json文件读取
def read_files_json(file_path):
    data = []
    for file in file_path:
        data.append(read_file_json(file))

    return data

# 查看指定路径下的文件
def read_folder(directory):
    files = []
    for filename in os.listdir(directory):
        if not os.path.isfile(os.path.join(directory, filename)):
            files.append(os.path.join(directory, filename))

    return files

def read_file(directory):
    files = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            files.append(os.path.join(directory, filename))

    return files

# 查看指定路径下的所有文件
def read_full_file(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    return files

# 读取指定路径下一定长度的json文件
def read_folder_files_json(directory, length = 0):
    Assert(length >= 0, "Error: length must be positive")

    files = read_folder(directory)
    if length > 0:
        files = files[0: length]

    data = read_files_json(files)

    return data


if __name__ == "__main__":
    json_folder_path = "D:/Study/Projects/DeepLearning/Resources/Indoor/3D-FRONT/"

    data = read_folder_files_json(json_folder_path, 10)

    print("Done!")