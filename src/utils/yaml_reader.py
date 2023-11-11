import yaml
import os

def read_yaml(path):
    # 读取 YAML 文件
    with open(path, "r") as file:
        yaml_data = yaml.safe_load(file)

    return yaml_data


def read_file_in_path(path, file_name):
    file_list = os.listdir(path)

    return read_yaml(os.path.join(path, file_name))
