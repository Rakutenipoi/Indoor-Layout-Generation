from torch.utils.data import DataLoader, SubsetRandomSampler
import tqdm
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from torch.cuda.amp import GradScaler, autocast

from network.network import cofs_network
from process.dataset import *
from utils.loss_cal import *
from utils.monitor import *

# 系统设置
#os.chdir(sys.path[0])

# pytorch设置
np.set_printoptions(suppress=True)
torch.backends.cudnn.benchmark = True
torch.set_default_dtype(torch.float32)

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
network_param = config['network']
training_config = config['training']
epochs = training_config['epochs']
batch_size = training_config['batch_size']
max_sequence_length = config['network']['max_sequence_length']
lr = training_config['lr']
checkpoint_freq = training_config['checkpoint_frequency']
dropout = training_config['dropout']

# wandb设置
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="cofs",
#
#     # track hyperparameters and run metadata
#     config={
#         "learning_rate": lr,
#         "architecture": "Transformer",
#         "dataset": "3D-Front",
#         "epochs": epochs,
#         "dropout": dropout,
#         "batch_size": batch_size,
#         "max_sequence_length": max_sequence_length,
#         "class_num": class_num,
#         "encoder_layers": network_param['n_enc_layers'],
#         "decoder_layers": network_param['n_dec_layers'],
#         "heads": network_param['n_heads'],
#         "dimensions": network_param['dimensions'],
#         "feed_forward_dimensions": network_param['feed_forward_dimensions'],
#         "activation": network_param['activation'],
#     }
# )

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
# 计算要读取的数据的数量
sample_size = int(0.1 * len(src_dataset))
# 创建一个随机抽样的索引列表
indices = torch.randperm(len(src_dataset))[:sample_size]
# 创建 SubsetRandomSampler，并传入索引列表
sampler = SubsetRandomSampler(indices)
dataLoader = DataLoader(src_dataset, batch_size=batch_size, num_workers=14, sampler=sampler)

# 创建网络模型
cofs_model = cofs_network(config).to(device)

# 从上次训练的模型参数开始训练
load_pretrain = False
model_param_path = 'model/full_data'
model_epoch_index = 0
# 读取该路径的文件
if (load_pretrain):
    model_param_files = os.listdir(model_param_path)
    # 从文件名bedrooms_model_后面的数字得到上次训练的epoch数
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

# 优化器
optimizer = torch.optim.AdamW(cofs_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
scaler = GradScaler()

if __name__ == '__main__':
    # tqdm
    pbar = tqdm.tqdm(total=epochs)
    print("dataLoader length: ", len(dataLoader))

    # 设置为训练模式
    cofs_model.train()

    # 迭代训练
    step = 0
    for epoch in range(model_epoch_index + 1, epochs):
        avg_loss = 0
        epoch_time = time.time()
        for batch in dataLoader:
            # 清除梯度
            optimizer.zero_grad(set_to_none=True)
            # 读取数据
            src_idx, layout_idx, seq_num = batch
            src_idx = src_idx.numpy().astype(int)
            seq_num = seq_num.numpy().astype(int)
            tgt_num = seq_num
            # 读取序列
            src = []
            for i in range(batch_size):
                src.append(sequences[src_idx[i]])
            src = np.array(src).reshape(batch_size, -1)
            src = torch.tensor(src, device=device)
            tgt = src.clone()
            # 读取布局
            layout = []
            layout_idx = layout_idx.numpy().astype(int)
            for i in range(batch_size):
                layout.append(layouts[layout_idx[i, 0]])
            layout = np.array(layout)
            layout = torch.tensor(layout, device=device, dtype=torch.get_default_dtype())
            # 设置requires_grad
            src.requires_grad = False
            layout.requires_grad = True
            tgt.requires_grad = True

            # # src随机mask
            # mask = torch.ones_like(src)
            # for i in range(batch_size):
            #     num = seq_num[i, 0]
            #     if num < 2:
            #         continue
            #     else:
            #         # 随机mask的数量为1到min(2, num)之间
            #         mask_min_num = 1
            #         mask_max_num = 2
            #         mask_rand_num = np.random.randint(mask_min_num, min(mask_max_num, num) + 1, 1)
            #         # 随机mask的位置
            #         mask_rand_pos = np.random.randint(0, num - 1, mask_rand_num)
            #         # mask
            #         for pos in mask_rand_pos:
            #             mask[i, pos * attr_num: (pos + 1) * attr_num] = 0
            # # mask与src对应位置相乘
            # src = src * mask

            # 设置requires_grad为True
            src.requires_grad = True

            # 前向传播
            output = cofs_model(src, layout, tgt, seq_num, tgt_num)
            # 计算损失
            loss = loss_calculate(src, output, tgt_num, config)

            #反向传播
            avg_loss += loss.item()
            loss.backward()

            # 更新参数
            optimizer.step()
            scheduler.step()

        # 更新pbar
        pbar.set_postfix(avg_loss=avg_loss / len(dataLoader), time=time.time() - epoch_time)
        pbar.update(1)

        if epoch % checkpoint_freq == 0:
            # 保存训练参数
            torch.save(cofs_model.state_dict(), model_param_path + f'/bedrooms_model_{epoch}.pth')
            print(f"Model saved at Epoch: {epoch}")

    wandb.finish()

