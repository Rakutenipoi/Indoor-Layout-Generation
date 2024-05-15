import sys

import numpy as np

from process.dataset import *
from network.network import cofs_network
from utils.draw import *
import time

# 系统设置
sys.setrecursionlimit(100000)
sys.stdout.flush = True

# pytorch设置
torch.set_printoptions(profile="full")
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

# GPU
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

# 读取config
config_path = '../config'
config_name = 'bedrooms_inference_config.yaml'
config = read_file_in_path(config_path, config_name)

# 数据集部分参数
attr_num = config['data']['attributes_num']
class_num = config['data']['class_num']
permutation_num = config['data']['permutation_num']
max_sequence_length = config['network']['max_sequence_length']

# 训练数据读取
data_path = '../data/processed/bedrooms/'
data_type = 'full_shuffled'
layouts = np.load(os.path.join(data_path, f'bedrooms_{data_type}_layout.npy'))

# 创建网络模型
cofs_model = cofs_network(config).to(device)

# 读取模型参数
model_param_path = '../model/sunny-snowflake-67'
# 读取该路径的文件
model_param_files = os.listdir(model_param_path)
# 从文件名bedrooms_model_后面的数字得到上次训练的epoch数
model_epoch_index = 0
for file in model_param_files:
    if file.startswith('bedrooms_model_'):
        file_epoch_index = int(file.split('.')[0].split('_')[-1])
        if file_epoch_index > model_epoch_index:
            model_epoch_index = file_epoch_index
model_param_name = f'bedrooms_model_{model_epoch_index}.pth'
model_param = torch.load(os.path.join(model_param_path, model_param_name))
cofs_model.load_state_dict(model_param)
print(f'load model {model_epoch_index}')

if __name__ == '__main__':
    # 设置为推理模式
    cofs_model.eval()

    # 推理次数
    inference_num = 5

    # 推理结果
    predicts = []
    layouts_for_inference = []
    # 根据inference_num从layouts中随机选取布局
    layout_size = layouts.shape[0]
    random_index = np.random.randint(0, layout_size, inference_num)
    inference_layouts = []

    for i in range(inference_num):
        # src序列
        layout = layouts[random_index[i]]
        inference_layouts.append(layout)
        layout = torch.tensor(layout, dtype=torch.float32)
        layout = layout.to(device)
        layout = layout.unsqueeze(0)
        src = torch.zeros((1, max_sequence_length), dtype=torch.float32).to(device)
        tgt = torch.zeros_like(src)
        seq_num = np.zeros((1, 1), dtype=np.int32)
        last_seq_num = np.zeros((1, 1), dtype=np.int32)

        for j in range(0, max_sequence_length, attr_num):
            output_type_prob, output_attr = cofs_model(src, layout, tgt, seq_num, last_seq_num)
            output_type = torch.softmax(output_type_prob, dim=-1)
            output_type = torch.argmax(output_type, dim=-1)

            src[0, j] = output_type[0, 0]
            src[0, j + 1 : j + attr_num] = output_attr[0, :, 0]

            tgt[0, j] = output_type[0, 0]
            tgt[0, j + 1 : j + attr_num] = output_attr[0, :, 0]

            seq_num[0, 0] += 1
            last_seq_num[0, 0] += 1

        predict = src.to('cpu').detach().numpy()
        predict = np.reshape(predict, (-1, attr_num))
        predicts.append(predict)
        layouts_for_inference.append(layouts[random_index[i]])

    # 推理结果整理
    predicts = np.array(predicts)
    sorted_seq = []
    index_seq = []
    max_len = 3
    for j in range(len(predicts)):
        predict = predicts[j]
        seq = []
        count = 0
        # predict_type记录已经出现过的家具类型
        predict_type = []
        for i in range(predict.shape[0]):
            if predict[i, 0] in predict_type or predict[i, 0] == 0:
                continue
            elif predict[i, 0] == 2 or predict[i, 0] == 8 or predict[i, 0] == 15:
                count += 1
                predict_type.append(predict[i, 0])
                seq.append(predict[i, :])

        if count == 0:
            continue

        for i in range(predict.shape[0]):
            if predict[i, 0] in predict_type or predict[i, 0] == 0:
                continue
            elif count > max_len:
                break
            else:
                count += 1
                predict_type.append(predict[i, 0])
                seq.append(predict[i, :])
        seq = np.array(seq)
        sorted_seq.append(seq)
        index_seq.append(j)

    # 打印推理结果
    for predict in sorted_seq:
        print(predict)
        print('---------------------------')

    sorted_layout = []
    for i in range(len(index_seq)):
        sorted_layout.append(inference_layouts[index_seq[i]])

    # 调用可视化函数
    for i in range(len(sorted_seq)):
        visiualize(sorted_layout[i], sorted_seq[i])