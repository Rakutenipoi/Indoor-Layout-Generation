import yaml

def read_yaml(path):
    # 读取 YAML 文件
    with open(path, "r") as file:
        yaml_data = yaml.safe_load(file)

    return yaml_data

