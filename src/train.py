import torch
import numpy as np
from torchviz import make_dot
import sys

from network.network import cofs_network
from utils.yaml_reader import read_file_in_path
from utils.loss import *
from process.read_dataset import *
from process.process_dataset import *

# 系统设置
sys.setrecursionlimit(100000)
sys.stdout.flush = True

# GPU
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

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
bNewData = False

if bNewData:
    src_data, layout_image = process_data(config)
    os.makedirs(os.path.dirname('data/processed/bedrooms/'), exist_ok=True)
    np.save('data/processed/bedrooms/bedrooms_src_data', src_data)
    np.save('data/processed/bedrooms/bedrooms_layout_image', layout_image)
else:
    src_data = np.load('data/processed/bedrooms/bedrooms_src_data.npy')
    layout_image = np.load('data/processed/bedrooms/bedrooms_layout_image.npy')

# 将numpy数据转换为tensor
src_data = torch.tensor(src_data, dtype=torch.float32)
layout_image = torch.tensor(layout_image, dtype=torch.float32)

## 其他处理
layout_image = layout_image.to(device)
src_data = src_data.to(device)
ground_truth = src_data.clone()

src_data.requires_grad = True
layout_image.requires_grad = True

# 其他数据
batch_num = src_data.size(0)

# 创建网络模型
cofs_model = cofs_network(config).to(device)
# print(cofs_model)

# 优化器
optimizer = torch.optim.Adam(cofs_model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

if __name__ == '__main__':
    # 设置为训练模式
    cofs_model.train()

    # 迭代训练
    for i in range(epoches):
        idx = i % batch_num
        src = src_data[idx]
        layout = layout_image[idx]
        tgt_data = torch.zeros(batch_size, max_sequence_length).to(device)
        tmp = torch.zeros(batch_size, max_sequence_length).to(device)

        for j in range(0, max_sequence_length, attr_num):
            # 前向传播
            output_type, output_attr = cofs_model(src, layout, tgt_data)

            # for param in cofs_model.named_parameters():
            #     print(param)

            ## 类型生成
            output_type_debug = torch.argmax(output_type, dim=2)
            gt_data_class = ground_truth[idx, :, j]
            ## 类别损失计算
            loss_type = class_type_loss_cross_nll(output_type.squeeze(1), gt_data_class)
            ## 其他属性生成
            loss_attr = property_loss_single(output_attr, ground_truth[idx, :, j + 1 : j + attr_num])

            # total loss
            #loss = (loss_type + loss_attr) / batch_size
            loss = loss_type / batch_size

            # 清除梯度
            optimizer.zero_grad()

            #反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # Teacher-forcing
            tmp[:, j: j + attr_num] = ground_truth[idx, :, j: j + attr_num]
            tgt_data = tmp.clone()
            tgt_data.requires_grad = True

            print(f"Epoch: {i+1}, token: {int(j / 8) + 1}, Loss: {loss.item() * 100.0}")

            # 可视化
            # dot = make_dot(loss, params=dict(cofs_model.named_parameters()))
            # dot.render('model/bedrooms_model', view=False)


    # 保存模型
    torch.save(cofs_model, 'model/bedrooms_model.pth')




