import torch
import numpy as np
from torchviz import make_dot
import sys
import msvcrt
import time
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
sequences = np.load('data/processed/bedrooms/bedrooms_full_sequence.npy')

# 提取数据
sequence_index, layout_index, sequences_num = extract_data(full_data)

# batch划分
batch_num = sequence_index.shape[0] // batch_size
sequence_index = sequence_index[:batch_num * batch_size]
layout_index = layout_index[:batch_num * batch_size]
sequences_num = sequences_num[:batch_num * batch_size]
layouts = layouts[:batch_num * batch_size]
sequences = sequences[:batch_num * batch_size]

# 数据集转换
src_dataset = COFSDataset(sequence_index, layout_index, sequences_num, config)
dataLoader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True)

# 创建网络模型
cofs_model = cofs_network(config).to(device)

# 优化器
optimizer = torch.optim.Adam(cofs_model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

if __name__ == '__main__':
    # 设置为训练模式
    cofs_model.train()

    # 迭代训练
    epoch_time_start = time.time()
    for epoch in range(epoches):
        batch_idx = 0
        epoch_time = time.time()
        batch_time_start = time.time()

        for batch in dataLoader:
            batch_time = time.time()
            batch_idx += 1
            print(f"batch: {batch_idx} / {dataLoader.__len__()}")
            # 读取数据
            src_idx, layout_idx, seq_num = batch
            src_idx = src_idx.numpy().astype(int)
            seq_num = seq_num.numpy().astype(int)
            # 读取序列
            src = []
            for i in range(batch_size):
                src.append(sequences[src_idx[i]])
            src = np.array(src).reshape(batch_size, -1)
            src = torch.tensor(src, dtype=torch.float32).to(device)
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

                # print per token
                print(f"token: {int(j / 8) + 1}, Loss: {loss.item()}")
                print("Type: ", end="")
                for type in output_type_debug:
                    print(type.item(), end=" ")
                print()

            # print per batch
            single_batch_time = time.time() - batch_time
            single_batch_time = "{:02d}:{:02d}:{:02d}".format(int(single_batch_time // 3600),
                                                              int((single_batch_time % 3600) // 60),
                                                              int(single_batch_time % 60))
            total_batch_time = time.time() - batch_time_start
            total_batch_time = "{:02d}:{:02d}:{:02d}".format(int(total_batch_time // 3600),
                                                             int((total_batch_time % 3600) // 60),
                                                             int(total_batch_time % 60))
            print(f"Time: {single_batch_time}, total time: {total_batch_time}")
            print('--------------------------------------------------------------------------------')

        # print per epoch
        print(f"Epoch: {epoch + 1} / {epoches}")
        single_epoch_time = time.time() - epoch_time
        single_epoch_time = "{:02d}:{:02d}:{:02d}".format(int(single_epoch_time // 3600), int((single_epoch_time % 3600) // 60), int(single_epoch_time % 60))
        total_epoch_time = time.time() - epoch_time_start
        total_epoch_time = "{:02d}:{:02d}:{:02d}".format(int(total_epoch_time // 3600), int((total_epoch_time % 3600) // 60), int(total_epoch_time % 60))
        print(f"Time: {single_epoch_time}, total time: {total_epoch_time}")
        print('--------------------------------------------------------------------------------')
        print('--------------------------------------------------------------------------------')

        if epoch % checkpoint_freq == 0:
            # 保存训练参数
            torch.save(cofs_model.state_dict(), f'model/bedrooms_model_{epoch + 1}.pth')
            print(f"Model saved at Epoch: {epoch + 1}")

    # 保存训练参数
    torch.save(cofs_model.state_dict(), 'model/bedrooms_model.pth')


