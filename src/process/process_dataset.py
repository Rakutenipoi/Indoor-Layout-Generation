import math
import os.path
import random
import numpy as np
import pandas as pd
import json
from src.process.read_dataset import *
from src.utils.yaml_reader import *
#from dataset import *
import itertools
import torch
from torch.utils.data import Dataset

# 整理原始数据，保留需要的部分
def original_full_data_filtering(oringal_data):
    filtered_data = []
    for house in oringal_data:
        needed_properties = ['uid', 'north_vector', 'furniture', 'extension', 'scene']
        data = original_house_data_filtering(house, needed_properties)

        filtered_data.append(data)

    return filtered_data

# 整理原始房屋数据
def original_house_data_filtering(house_data, needed_properties):
    data = {}
    for needed_property in needed_properties:
        data[needed_property] = house_data[needed_property]

    return data

# 提取卧室数据
def extract_bedroom_data(original_data):
    scenes_bedroom_data = []
    spected_room_type = ['KidsRoom', 'Bedroom', 'MasterBedroom', 'SecondBedroom']
    for house in original_data:
        scene = house['scene']
        rooms = scene['room']
        scenes_bedroom_data.append(extract_spected_room(rooms, spected_room_type))

    return scenes_bedroom_data

# 特定房间类型提取
def extract_spected_room(rooms, spected_room_type):
    spected_room = []
    for room in rooms:
        type = room['type']
        if type in spected_room_type:
            spected_room.append(room)

    return spected_room

# 提取出卧室场景的全部家具类型
def extract_classes_from_rooms(scenes):
    furniture_class = []

    for scene in scenes:
        for room in scene:
            furniture_class.extend(extract_furniture(room))

    return furniture_class

# 家具类型提取
def extract_furniture(room):
    children = room['children']
    furnitures = []

    for child in children:
        if 'furniture' in child['instanceid']:
            furnitures.append(child)

    return furnitures

def filter_repeat_classes(class_data):
    class_set = set()
    unique_category = [x for x in class_data if not (x['ref'] in class_set or class_set.add(x['ref']))]

    return unique_category

def build_data_set(rooms):
    pass

def process_data(config):
    max_sequence_length = config['network']['max_sequence_length']
    batch_size = config['training']['batch_size']
    object_max_num = config['data']['object_max_num']
    class_num = config['data']['class_num']
    permutation_num = config['data']['permutation_num']
    bounds_translations = config['data']['bounds_translations']
    bounds_sizes = config['data']['bounds_sizes']
    bounds_angles = config['data']['bounds_angles']

    data_folder = read_folder(config['data']['dataset_directory'])
    rooms_dir = []
    for folder in data_folder:
        rooms_dir.append(read_file(folder))

    rooms = []
    layouts = []
    index = []
    idx_layout = 0
    idx_sequence = 0
    scene_dict = []

    for room_dir in rooms_dir:
        furnitures = []
        room = np.load(room_dir[0], allow_pickle=True)
        class_labels = room['class_labels'][:, :class_num]
        translations = room['translations']
        sizes = room['sizes']
        angles = room['angles']
        layout = room['room_layout']

        for i in range(class_labels.shape[0]):
            furniture = []

            if class_labels[i].sum() == 0:
                continue

            # 读取(1, 23)的nparray中具有最大值的元素索引
            class_label = np.argmax(class_labels[i]) + 1
            translation = translations[i]
            size = sizes[i]
            angle = angles[i]

            # 数据归一化
            for j in range(3):
                translation[j] = (translation[j] - bounds_translations[j]) / (bounds_translations[j + 3] - bounds_translations[j])
                size[j] = (size[j] - bounds_sizes[j]) / (bounds_sizes[j + 3] - bounds_sizes[j])
            angle[0] = (angle[0] - bounds_angles[0]) / (bounds_angles[1] - bounds_angles[0])
            if angle[0] < 0 or translation[0] < 0 or translation[1] < 0 or translation[2] < 0 or size[0] < 0 or size[1] < 0 or size[2] < 0:
                continue
            elif angle[0] > 1 or translation[0] > 1 or translation[1] > 1 or translation[2] > 1 or size[0] > 1 or size[1] > 1 or size[2] > 1:
                continue

            furniture.append(class_label)
            furniture.append(translation[0])
            furniture.append(translation[1])
            furniture.append(translation[2])
            furniture.append(size[0])
            furniture.append(size[1])
            furniture.append(size[2])
            furniture.append(angle[0])
            furnitures.append(furniture)

        furniture_num = len(furnitures)
        if furniture_num <= 1:
            continue
        elif furniture_num > object_max_num:
            furnitures = furnitures[:object_max_num]
        # 对furnitures进行shuffle
        random.shuffle(furnitures)
        # 根据object_max_num进行填充，得到(object_max_num, 8)的nparray
        while len(furnitures) < object_max_num:
            furnitures.append([0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        furnitures = np.array(furnitures)
        rooms.append(furnitures)
        layouts.append(layout)
        index.append([idx_layout, idx_sequence, furniture_num])
        idx_layout += 1
        idx_sequence += 1

    layouts = np.array(layouts)
    rooms = np.array(rooms)
    index = np.array(index, dtype=object)
    #rooms = np.expand_dims(rooms, axis=1)

    return rooms, layouts, index, scene_dict

# 返回所有可能的家具排序
def permute_furniture(rooms):
    src_data, layout_image = extract_data(rooms)
    extended_data = []
    # 由于是对同一个房间内的家具排序进行修改，故不用变更layout中的数据排序
    for i in range(src_data.shape[0]):
        data = src_data[i, 0]
        # 先获取房间的家具数量
        zero_row_indices = np.where((data == 0).all(axis=1))[0][0]
        permutations = list(itertools.permutations(range(zero_row_indices)))

        full_sorted_data = []
        for perm in permutations:
            sorted_data = []
            sorted_data.append(np.zeros_like(data))
            sorted_data[-1][:zero_row_indices] = data[list(perm)]
            full_sorted_data.append(sorted_data)
        full_sorted_data = np.array(full_sorted_data)
        full_sorted_data = np.squeeze(full_sorted_data, axis=1)
        extended_data.append([full_sorted_data, layout_image[i]])

    extended_data = np.array(extended_data, dtype=object)

    return extended_data

def permute(length):
    permutations = list(itertools.permutations(range(length)))

    return permutations

# 从object类型的ndarray中提取家具序列和布局图
def extract_data(rooms):
    # rooms = rooms.reshape(-1, 3)
    layout = np.array(rooms[:, 0])
    sequence = np.array(rooms[:, 1])
    length = np.array(rooms[:, 2])

    original_shape = sequence.shape
    new_shape_src = np.empty((original_shape[0], 1))
    new_shape_layout = np.empty((original_shape[0], 1))
    new_shape_length = np.empty((original_shape[0], 1))
    for i in range(original_shape[0]):
        new_shape_src[i, :] = sequence[i]
        new_shape_layout[i, :] = layout[i]
        new_shape_length[i, :] = length[i]

    return new_shape_src, new_shape_layout, new_shape_length

def shuffle_data(rooms):
    for i in range(rooms.shape[0]):
        data = rooms[i]
        zero_row_indices = np.where((data == 0).all(axis=1))[0][0]
        np.random.shuffle(data[:zero_row_indices])

    return rooms

if __name__ == "__main__":
    # json_folder_path = "D:/Study/Projects/DeepLearning/Resources/Indoor/3D-FRONT/"
    #
    # # 读取原始数据
    # original_data = read_folder_files_json(json_folder_path, 50)
    # # 剔除原始数据中不需要的属性
    # data = original_full_data_filtering(original_data)
    # # 提取出卧室场景的数据
    # scenes_bedroom_data = extract_bedroom_data(data)
    # # 提取出卧室场景的全部家具类型 (需要注意，此时的家具可能包含重复的类型)
    # furniture_class = extract_classes_from_rooms(scenes_bedroom_data)
    # # 类型去重
    # unique_furniture_class = filter_repeat_classes(furniture_class)
    # class_num = len(unique_furniture_class)
    #
    # # 读取模型信息表
    # # model_info_path = "D:/Study/Projects/DeepLearning/Resources/Indoor/3D-FUTURE-model"
    # # model_info = read_file_json(os.path.join(model_info_path, "model_info.json"))
    #
    # # 构建数据集
    # rooms = []
    # for scene in scenes_bedroom_data:
    #     for room in scene:
    #         rooms.append(room)
    # data_set = build_data_set(rooms)
    # 读取config
    config_path = '../../config'
    config_name = 'bedrooms_config.yaml'
    config = read_file_in_path(config_path, config_name)
    rooms, layouts, index, scene_dict = process_data(config)
    # rooms = permute_furniture(rooms[:10])

    data_path = '../../data/processed/bedrooms/'
    usage = 'full_shuffled'
    file_name = f'bedrooms_{usage}_sequence'
    layout_name = f'bedrooms_{usage}_layout'
    data_name = f'bedrooms_{usage}_data'
    # os.makedirs(os.path.dirname(data_path), exist_ok=True)
    # np.save(os.path.join(data_path, file_name), rooms)
    # np.save(os.path.join(data_path, layout_name), layouts)
    # np.save(os.path.join(data_path, data_name), index)

    #src_data, layout_image = extract_data(rooms)

    # 将scene_dict保存为json文件
    scene_dict_name = 'dataset'
    scene_dict_path = os.path.join(data_path, scene_dict_name)
    with open(scene_dict_path, 'w') as file:
        json.dump(scene_dict, file)

    print("done")
