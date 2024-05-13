import os
import json
from shutil import copyfile
from read_dataset import *
import numpy as np
import tqdm
from PIL import Image
import pandas as pd

from src.utils.yaml_reader import read_file_in_path

# 读取配置文件
config_path = '../../config'
config_name = 'bedrooms_config.yaml'
config = read_file_in_path(config_path, config_name)
bounds_translations = config['data']['bounds_translations']
bounds_sizes = config['data']['bounds_sizes']
bounds_angles = config['data']['bounds_angles']

# 定义原始文件夹和新文件夹的路径
raw_folder = config['data']['dataset_directory']
new_folder = config['data']['processed_directory']

# 处理json文件
def npz2json(path):
    room = np.load(path, allow_pickle=True)
    furnitures = []
    class_labels = room['class_labels']
    translations = room['translations']
    sizes = room['sizes']
    angles = room['angles']

    for i in range(class_labels.shape[0]):
        furniture = {}

        if class_labels[i].sum() == 0:
            continue

        # 读取(1, 23)的nparray中具有最大值的元素索引
        class_label = np.argmax(class_labels[i]) + 1
        translation = translations[i]
        size = sizes[i]
        angle = angles[i]


        # 数据归一化
        for j in range(3):
            translation[j] = (translation[j] - bounds_translations[j]) / (
                        bounds_translations[j + 3] - bounds_translations[j])
            size[j] = (size[j] - bounds_sizes[j]) / (bounds_sizes[j + 3] - bounds_sizes[j])
        angle[0] = (angle[0] - bounds_angles[0]) / (bounds_angles[1] - bounds_angles[0])
        if angle[0] < 0 or translation[0] < 0 or translation[1] < 0 or translation[2] < 0 or size[0] < 0 or size[
            1] < 0 or size[2] < 0:
            continue
        elif angle[0] > 1 or translation[0] > 1 or translation[1] > 1 or translation[2] > 1 or size[0] > 1 or size[
            1] > 1 or size[2] > 1:
            continue

        furniture['class'] = class_label.tolist()
        furniture['translation'] = translation.tolist()
        furniture['size'] = size.tolist()
        furniture['angle'] = angle[0].tolist()
        furnitures.append(furniture)

    return furnitures

if __name__ == "__main__":
    # 创建新文件夹
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    # 创建tqdm进度条
    pbar = tqdm.tqdm(total=len(os.listdir(raw_folder)))

    # 遍历原始文件夹中的每个子文件夹
    png_info = []
    png_data = []
    for subdir in os.listdir(raw_folder):
        pbar.update(1)
        subdir_path = os.path.join(raw_folder, subdir)
        if os.path.isdir(subdir_path):
            # 初始化存储 PNG 文件名和信息的列表
            npz_path = ''
            # 遍历子文件夹中的每个文件
            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                # 如果是 PNG 文件，则复制到新文件夹
                if filename.endswith('.png'):
                    new_png_path = os.path.join(new_folder, f"{subdir}.png")
                    img = Image.open(file_path)
                    copyfile(file_path, new_png_path)
                else:
                    npz_path = file_path
                    json_objects = npz2json(file_path)
                    room = np.load(file_path, allow_pickle=True)
                    layout = room['room_layout']
            # 存储文件信息
            png_info.append({
                "file_name": f"{subdir}.png",
                "objects": json_objects,
                "layout": layout
            })
            png_data.append({
                "file_name": f"{subdir}.png",
                "layout": img.tobytes(),
                "objects": json_objects
            })

    # 生成存储 PNG 文件信息的 JSON 文件
    # json_path = os.path.join(new_folder, "metadata.jsonl")
    # with open(json_path, 'w') as f:
    #     json.dump(png_info, f)

    # 保存为pandas DataFrame
    df = pd.DataFrame(png_info)
    df.to_pickle(os.path.join(new_folder, "metadata.pkl"))

    print("done")
