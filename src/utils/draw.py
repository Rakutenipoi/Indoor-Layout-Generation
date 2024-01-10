import cv2
import os
import numpy as np
from src.utils.yaml_reader import *

# 读取config
config_path = '../config'
config_name = 'class_map.yaml'
config = read_file_in_path(config_path, config_name)['class_order']

def visiualize(layout, sequence):
    # 获得layout的尺寸
    # 根据二值图中包围白色像素点的最小矩形，得到家具的边界
    ## 查找白色像素点的轮廓
    contours, _ = cv2.findContours(layout, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ## 获取最小矩形的边界框
    l_x, l_z, l_w, l_h = cv2.boundingRect(contours[0])
    width = l_w
    height = l_h

    bounds_translations = [-2.762500499999998, 0.045, -2.7527500000000007, 2.778441746198965, 3.6248395981292725,
                           2.818542771063899]
    bounds_sizes = [0.0399828836528625, 0.020000020334800084, 0.012771999999999964, 2.8682, 1.7700649999999998,
                    1.698315]
    bounds_angles = [-3.1416, 3.1416]

    # 绘制家具布局俯视图
    size_ratio = 50
    img = np.copy(layout)
    # 将img转换为三通道
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    furniture_type = []
    for i in range(sequence.shape[0]):
        if sequence[i, 0] == 21:
            continue
        else:
            x = l_x + int(sequence[i, 1] * width)
            z = l_z + int(sequence[i, 3] * height)
            w = int((bounds_sizes[0] + sequence[i, 4] * (bounds_sizes[3] - bounds_sizes[0])) * size_ratio)
            h = int((bounds_sizes[2] + sequence[i, 6] * (bounds_sizes[5] - bounds_sizes[2])) * size_ratio)
            furniture_type.append((int(sequence[i, 0]), config.get(int(sequence[i, 0]))))
            # 矩形参数
            angle = int((sequence[i, 7] * (bounds_angles[1] - bounds_angles[0]) + bounds_angles[0]) / np.pi * 180)
            center = (x, z)
            pts = np.array([[x - w / 2, z - h / 2], [x + w / 2, z - h / 2], [x + w / 2, z + h / 2], [x - w / 2, z + h / 2]], np.int32)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_pts = cv2.transform(np.array([pts]), rotation_matrix)[0]
            cv2.polylines(img, [rotated_pts], isClosed=True, color=(0, 0, 255), thickness=1)
            cv2.putText(img, str(int(sequence[i, 0])), (x, z), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    resize_ratio = 4
    new_size = (resize_ratio * layout.shape[1], resize_ratio * layout.shape[0])
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    furniture_img = np.zeros((new_size[0], new_size[1] + 200, 3), np.uint8)
    # 将resized_img复制到furniture_img中
    furniture_img[0:new_size[0], 0:new_size[1]] = resized_img
    type_put_pos = (int(resize_ratio * (l_x + width + 10)), int(resize_ratio * (l_z + height * 3 / 4)))
    for i in range(len(furniture_type)):
        text = str(furniture_type[i][0]) + ': ' + furniture_type[i][1]
        cv2.putText(furniture_img, text, type_put_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        type_put_pos = (type_put_pos[0], type_put_pos[1] + 40)
    cv2.imshow('furnitures', furniture_img)

    # 布局二值图展示
    cv2.imshow('layout', layout)
    key = cv2.waitKey(0)

    # 布局二值图保存
    inference_path = '../inference'
    good_path = 'good'
    medium_path = 'medium'
    bad_path = 'bad'
    if key == ord('1'):
        inference_path = os.path.join(inference_path, good_path)
    elif key == ord('2'):
        inference_path = os.path.join(inference_path, medium_path)
    else:
        inference_path = os.path.join(inference_path, bad_path)

    files = os.listdir(os.path.join(inference_path, 'layout'))

    # 从文件名f'{index}_layout.jpg'的index得到最大的index值
    index = 0
    for file in files:
        if file.endswith('_layout.jpg'):
            file_index = int(file.split('_')[0])
            if file_index >= index:
                index = file_index + 1

    cv2.imwrite(inference_path + f'/layout/{index}_layout.jpg', layout)
    cv2.imwrite(inference_path + f'/furnitures/{index}_furnitures.png', furniture_img)

    # 家具序列保存
    with open(inference_path + f'/sequence/{index}_sequence.txt', 'w') as f:
        for i in range(sequence.shape[0]):
            for j in range(sequence.shape[1]):
                f.write(str(sequence[i, j]) + ' ')
            f.write('\n')