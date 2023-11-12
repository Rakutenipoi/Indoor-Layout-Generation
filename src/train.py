import torch
import numpy as np
import sys
from torchview import draw_graph

from network.network import cofs_network
from utils.yaml_reader import read_file_in_path
from utils.loss import property_loss_distribution, property_loss_single, class_type_loss

# GPU
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# 系统设置
# sys.setrecursionlimit(10000)

# 读取config
config_path = 'config'
config_name = 'bedrooms_config.yaml'
config = read_file_in_path(config_path, config_name)

# 数据集部分参数
attr_num = config['data']['attributes_num']
class_num = config['data']['class_num']

# 训练数据设置
training_config = config['training']
epoches = training_config['epoches']
batch_size = training_config['batch_size']
max_sequence_length = config['network']['max_sequence_length']
lr = training_config['lr']

# 训练数据读取
num_elements = max_sequence_length
bNewData = True

## 二值图
layout_image = torch.randn(batch_size, 1, 256, 256)

## src
if bNewData:
    src_data = torch.rand(batch_size, num_elements, 1)
    for i in range(src_data.size()[1]):
        if i % attr_num == 0:
            src_data[:, i, :] = src_data[:, i, :] * 100
    src_list = src_data.numpy()
    np.save('data/src.npy', src_list)
    src_data = torch.tensor(src_data)
else:
    src_data = torch.from_numpy(np.load('data/src.npy'))

## tgt
tgt_data = torch.zeros_like(src_data)

## 其他处理
layout_image = layout_image.to(device)
src_data = src_data.to(device)
tgt_data = tgt_data.to(device)

# 创建网络模型
cofs_model = cofs_network(config).to(device)

# 优化器
optimizer = torch.optim.Adam(cofs_model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

# 模型可视化
# make_dot(output, params=dict(list(cofs_model.named_parameters()))).render("model_view/type_cofs_transformer_torchviz", format="png")

if __name__ == '__main__':
    # 设置为训练模式
    cofs_model.train()

    # 迭代训练
    for i in range(epoches):
        for j in range(0, max_sequence_length, attr_num):
            # 前向传播
            output_type, output_attr = cofs_model(src_data, layout_image, tgt_data)
            ## 类型生成
            output_type = torch.argmax(output_type, dim=2)
            src_data_class = src_data[:, j, :]
            loss_type = class_type_loss(output_type, src_data_class)
            ## 其他属性生成
            loss_attr = property_loss_single(output_attr, src_data[:, j : j + attr_num - 1, :])

            # total loss
            loss = (loss_type + loss_attr) / batch_size

            # Teacher-forcing
            tgt_data[:, j : j + attr_num, :] = src_data[:, j : j + attr_num, :]

            # 清除梯度
            optimizer.zero_grad()

            #反向传播
            loss.backward()

            # 更新参数
            optimizer.step()
            print(f"Epoch: {i+1}, token: {int(j / 8) + 1}, Loss: {loss.item()}")




    print('done')