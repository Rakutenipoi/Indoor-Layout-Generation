import json
import os

# json文件读取
def read_file_json(file_path):
    assert os.path.exists(file_path), "ERROR: JSON file path doesn't exist."
    with open(file_path, 'r') as file:
        data = json.load(file)

    return data

# 查看指定路径下的文件
def read_folder(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))

    return files

if __name__ == "__main__":
    json_folder_path = "D:/Study/Projects/DeepLearning/Resources/Indoor/3D-FRONT/"
    files = read_folder(json_folder_path)
    json_data = read_file_json(files[0])

    print("Done!")