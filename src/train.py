import torch
import numpy as np

from network.network import cofs_network
from utils.yaml_reader import read_file_in_path
from utils.loss import property_loss_distribution, property_loss_single, class_type_loss

# GPU
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# 读取config
config_path = 'config'
config_name = 'bedrooms_config.yaml'
config = read_file_in_path(config_path, config_name)

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
        if i % config['data']['attributes_num'] == 0:
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

if __name__ == '__main__':
    # 设置为训练模式
    cofs_model.train()

    # 迭代训练
    for i in range(epoches):
        for j in range(max_sequence_length):
            # 前向传播
            if j % 8 == 0:
                output = cofs_model(src_data, layout_image, tgt_data, True)
                output = torch.argmax(output, dim=2)
                loss = class_type_loss(output, src_data[:, j, :])
            else:
                output = cofs_model(src_data, layout_image, tgt_data, False)
                loss = property_loss_single(output, src_data[:, j, :])

            # Teacher-forcing
            tgt_data[:,j,:] = src_data[:,j,:]

            # 清除梯度
            optimizer.zero_grad()

            #反向传播
            loss.backward()

            # 更新参数
            optimizer.step()
            print(f"Epoch: {i+1}, Loss: {loss.item()}")




    print('done')