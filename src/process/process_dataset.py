import os.path
import numpy as np
from .read_dataset import *
from src.utils.yaml_reader import *

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

    data_folder = read_folder(config['data']['dataset_directory'])
    rooms_dir = []
    for folder in data_folder:
        rooms_dir.append(read_file(folder))

    rooms = []
    for room_dir in rooms_dir:
        sequence = []
        room = np.load(room_dir[0], allow_pickle=True)
        class_labels = room['class_labels']
        translations = room['translations']
        sizes = room['sizes']
        angles = room['angles']
        layout = room['room_layout']
        for i in range(class_labels.shape[0]):
            # 读取(1, 23)的nparray中具有最大值的元素索引
            class_label = np.argmax(class_labels[i])
            translation = translations[i]
            size = sizes[i]
            angle = angles[i]
            sequence.append(class_label)
            sequence.append(translation[0])
            sequence.append(translation[1])
            sequence.append(translation[2])
            sequence.append(size[0])
            sequence.append(size[1])
            sequence.append(size[2])
            sequence.append(angle[0])

        # 根据max_sequence_length进行填充
        while len(sequence) < max_sequence_length:
            sequence.append(0)
        sequence = np.array(sequence)
        rooms.append([sequence, layout])

    # 将rooms按照batch_size进行划分
    batch_num = int(len(rooms) / batch_size)
    rooms = np.array(rooms, dtype=object)[:batch_num * batch_size]
    rooms = rooms.reshape(-1, batch_size, 2)
    # 将rooms中的数据分别存储到src_data和layout_image
    sequence = np.array(rooms[:, :, 0])
    layout = np.array(rooms[:, :, 1])
    original_shape = sequence.shape
    new_shape_src = np.empty((original_shape[0], original_shape[1], sequence[0, 0].shape[0]))
    new_shape_layout = np.empty((original_shape[0], original_shape[1], layout[0, 0].shape[0], layout[0, 0].shape[1]))
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            new_shape_src[i, j, :] = sequence[i][j]
            new_shape_layout[i, j, :, :] = np.squeeze(layout[i][j])

    return new_shape_src, new_shape_layout

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
    process_data(config)

    print("done")
