from read_dataset import read_folder_files_json
from src.utils.debug_message import Assert

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
    bedroom_data = []
    spected_room_type = ['KidsRoom', 'Bedroom', 'MasterBedroom', 'SecondBedroom']
    for house in original_data:
        scene = house['scene']
        rooms = scene['room']
        bedroom_data.append(extract_spected_room(rooms, spected_room_type))

    return bedroom_data

# 特定房间类型提取
def extract_spected_room(rooms, spected_room_type):
    spected_room = []
    for room in rooms:
        type = room['type']
        if type in spected_room_type:
            spected_room.append(room)

    return spected_room

if __name__ == "__main__":
    json_folder_path = "D:/Study/Projects/DeepLearning/Resources/Indoor/3D-FRONT/"

    original_data = read_folder_files_json(json_folder_path, 10)
    data = original_full_data_filtering(original_data)

    bedroom_data = extract_bedroom_data(data)

    print("done")
