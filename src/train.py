import torch
import numpy as np
import sys
import time
from torch.utils.data import DataLoader
import tqdm

from network.network import cofs_network
from utils.yaml_reader import read_file_in_path
from utils.loss import *
from process.read_dataset import *
from process.process_dataset import *
from process.dataset import *
from utils.loss_cal import *

# 系统设置
#os.chdir(sys.path[0])

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
config_path = 'config'
config_name = 'bedrooms_config.yaml'
config = read_file_in_path(config_path, config_name)

# 数据集部分参数
attr_num = config['data']['attributes_num']
class_num = config['data']['class_num']
permutation_num = config['data']['permutation_num']

# 训练数据设置
training_config = config['training']
epoches = training_config['epoches']
batch_size = training_config['batch_size']
max_sequence_length = config['network']['max_sequence_length']
lr = training_config['lr']
checkpoint_freq = training_config['checkpoint_frequency']

# 训练数据读取
data_path = 'data/processed/bedrooms/'
data_type = 'full_shuffled'
full_data = np.load(os.path.join(data_path, f'bedrooms_{data_type}_data.npy'), allow_pickle=True)
layouts = np.load(os.path.join(data_path, f'bedrooms_{data_type}_layout.npy'))
sequences = np.load(os.path.join(data_path, f'bedrooms_{data_type}_sequence.npy'))

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

# 从上次训练的模型参数开始训练
load_pretrain = False
model_param_path = 'model/full_data'
# 读取该路径的文件
if (load_pretrain):
    model_param_files = os.listdir(model_param_path)
    # 从文件名bedrooms_model_后面的数字得到上次训练的epoch数
    model_epoch_index = 0
    for file in model_param_files:
        if file.startswith('bedrooms_model_'):
            file_epoch_index = int(file.split('.')[0].split('_')[-1])
            if file_epoch_index > model_epoch_index:
                model_epoch_index = file_epoch_index

    # 读取模型参数
    if len(model_param_files) > 0:
        model_param_name = f'bedrooms_model_{model_epoch_index}.pth'
        model_param = torch.load(os.path.join(model_param_path, model_param_name))
        cofs_model.load_state_dict(model_param)

# 打印模型
#print(cofs_model)

# 优化器
optimizer = torch.optim.Adam(cofs_model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

if __name__ == '__main__':
    # tqdm
    pbar = tqdm.tqdm(total=epoches * dataLoader.__len__())

    # 设置为训练模式
    cofs_model.train()

    # 迭代训练
    for epoch in range(model_epoch_index + 1, epoches):
        for batch in dataLoader:
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
            layout = np.array(layout)
            layout = torch.tensor(layout, dtype=torch.float32).to(device)
            # 设置requires_grad
            src.requires_grad = False
            layout.requires_grad = True
            tgt.requires_grad = True
            last_seq_num = np.zeros_like(seq_num)

            # src随机mask
            mask = torch.ones_like(src)
            for i in range(batch_size):
                num = seq_num[i, 0]
                if num < 2:
                    continue
                else:
                    # 随机mask的数量为1到min(2, num)之间
                    mask_min_num = 1
                    mask_max_num = 2
                    mask_rand_num = np.random.randint(mask_min_num, min(mask_max_num, num) + 1, 1)
                    # 随机mask的位置
                    mask_rand_pos = np.random.randint(0, num - 1, mask_rand_num)
                    # mask
                    for pos in mask_rand_pos:
                        mask[i, pos * attr_num: (pos + 1) * attr_num] = 0
            # mask与src对应位置相乘
            src = src * mask

            # 计算seq_len的平均值
            seq_len_sum = 0
            for i in range(batch_size):
                seq_len_sum += seq_num[i, 0]
            seq_len_mean = seq_len_sum / batch_size
            # 向上取整
            seq_len_mean = int(np.ceil(seq_len_mean))

            # 设置requires_grad为True
            src.requires_grad = True

            for j in range(0, seq_len_mean * attr_num, attr_num):
                # if src[:, j].sum() == 0:
                #     break

                # # 计算每个batch中类别为0的个数
                # zero_class_num = 0
                # for i in range(batch_size):
                #     if src[i, j] == 0:
                #         zero_class_num += 1
                # # 如果类别为0的个数大于等于batch_size的一半，则认为这个batch中的所有序列都已经结束
                # if zero_class_num >= batch_size // 2:
                #     break

                # 前向传播
                output_type, output_attr = cofs_model(src, layout, tgt, seq_num, last_seq_num)

                # for param in cofs_model.named_parameters():
                #     print(param)

                ## 类型生成
                output_type_debug = torch.argmax(output_type, dim=2)
                gt_data_class = ground_truth[:, j]
                ## 类别损失计算
                #loss_type = dmll(output_type, gt_data_class.unsqueeze(-1).unsqueeze(-1), num_classes=class_num)
                loss_type = class_type_loss_cross_nll(output_type.squeeze(1), gt_data_class)
                ## 其他属性生成
                #loss_attr = dmll(output_attr, ground_truth[:, j + 1: j + attr_num].unsqueeze(-1))
                loss_attr = property_loss_single(output_attr.squeeze(), ground_truth[:, j + 1: j + attr_num])
                loss_translation = torch.sum(loss_attr[:, :3], dim=-1)
                loss_size = torch.sum(loss_attr[:, 3:6], dim=-1)
                loss_rotation = torch.sum(loss_attr[:, 6:], dim=-1)

                # total loss
                loss_per_batch = loss_type + loss_translation + loss_size + loss_rotation
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

                # set last_seq_num
                for i in range(batch_size):
                    if output_type_debug[i, 0] != class_num + 2 - 1:
                        last_seq_num[i] += 1

            # pbar.set_postfix(loss=)
            pbar.update(1)

        if epoch % checkpoint_freq == 0:
            # 保存训练参数
            torch.save(cofs_model.state_dict(), model_param_path + f'/bedrooms_model_{epoch}.pth')
            print(f"Model saved at Epoch: {epoch}")



