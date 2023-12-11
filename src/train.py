import torch
import numpy as np
from torchviz import make_dot
import sys
import msvcrt
from torch.utils.data import DataLoader

from network.network import cofs_network
from utils.yaml_reader import read_file_in_path
from utils.loss import *
from process.read_dataset import *
from process.process_dataset import *
from process.dataset import *

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
checkpoint_freq = training_config['checkpoint_frequency']

# 训练数据读取
full_data = np.load('data/processed/bedrooms/bedrooms_full_data.npy', allow_pickle=True)
layouts = np.load('data/processed/bedrooms/bedrooms_full_layout.npy')

src_data, layout_index = extract_data(full_data)
src_data = shuffle_data(src_data)
src_data = src_data.reshape((src_data.shape[0], -1))
batch_num = src_data.shape[0] // batch_size
src_data = src_data[:batch_num * batch_size]
layout_index = layout_index[:batch_num * batch_size]
layouts = layouts[:batch_num * batch_size]

# 数据集转换
src_dataset = COFSDataset(src_data, layout_index)
dataLoader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True)

# 创建网络模型
cofs_model = cofs_network(config).to(device)

# 优化器
optimizer = torch.optim.Adam(cofs_model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

if __name__ == '__main__':
    # 设置为训练模式
    cofs_model.train()

    # 迭代训练
    for epoch in range(epoches):
        print(f"Epoch: {epoch + 1} / {epoches}")
        batch_idx = 0
        for batch in dataLoader:
            batch_idx += 1
            print(f"batch: {batch_idx} / {dataLoader.__len__()}")
            # 读取数据
            src, layout_idx = batch
            src = src.to(torch.float32).to(device)
            ground_truth = src.clone()
            tmp = torch.zeros_like(ground_truth)
            tgt = torch.zeros_like(ground_truth)
            # 读取布局
            layout = []
            layout_idx = layout_idx.numpy().astype(int)
            for i in range(batch_size):
                layout.append(layouts[layout_idx[i, 0]])
            layout = torch.tensor(layout, dtype=torch.float32).to(device)
            # 设置requires_grad
            src.requires_grad = True
            layout.requires_grad = True
            tgt.requires_grad = True

            for j in range(0, max_sequence_length, attr_num):
                # 前向传播
                output_type, output_attr = cofs_model(src, layout, tgt)

                # for param in cofs_model.named_parameters():
                #     print(param)

                ## 类型生成
                output_type_debug = torch.argmax(output_type, dim=2)
                gt_data_class = ground_truth[:, j]
                ## 类别损失计算
                loss_type = dmll(output_type, gt_data_class.unsqueeze(-1).unsqueeze(-1), num_classes=class_num)
                ## 其他属性生成
                loss_attr = dmll(output_attr, ground_truth[:, j + 1: j + attr_num].unsqueeze(-1))
                loss_attr_sum = torch.sum(loss_attr, dim=-1)

                # total loss
                loss_per_batch = loss_type.squeeze() + loss_attr_sum
                loss = torch.sum(loss_per_batch) / batch_size

                # 清除梯度
                optimizer.zero_grad()

                #反向传播
                loss.backward()

                # 更新参数
                optimizer.step()

                # Teacher-forcing
                tmp[:, j: j + attr_num] = ground_truth[:, j: j + attr_num]
                tgt = tmp.clone()
                tgt.requires_grad = True
                print(f"token: {int(j / 8) + 1}, Loss: {loss.item()}")
                print("Type: ", end="")
                for type in output_type_debug:
                    print(type.item(), end=" ")
                print()

        if epoch % checkpoint_freq == 0:
            # 保存训练参数
            torch.save(cofs_model.state_dict(), f'model/bedrooms_model_{epoch + 1}.pth')
            print(f"Model saved at Epoch: {epoch + 1}")

    # 保存训练参数
    torch.save(cofs_model.state_dict(), 'model/bedrooms_model.pth')


