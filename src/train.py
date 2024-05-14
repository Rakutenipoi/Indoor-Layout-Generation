from torch.utils.data import DataLoader
import tqdm
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
import cProfile
import re

from network.network import cofs_network
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

# 性能分析
cProfile.run('re.compile("foo|bar")')

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
epochs = training_config['epochs']
batch_size = training_config['batch_size']
max_sequence_length = config['network']['max_sequence_length']
lr = training_config['lr']
checkpoint_freq = training_config['checkpoint_frequency']

# wandb设置
wandb.init(
    # set the wandb project where this run will be logged
    project="cofs",

    # track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "architecture": "Transformer",
        "dataset": "3D-Front",
        "epochs": epochs,
    }
)

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
optimizer = torch.optim.Adam(cofs_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

if __name__ == '__main__':
    # tqdm
    pbar = tqdm.tqdm(total=epochs * dataLoader.__len__())

    # 设置为训练模式
    cofs_model.train()

    # 迭代训练
    for epoch in range(model_epoch_index + 1, epochs):
        for batch in dataLoader:
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
            src = torch.tensor(src, dtype=torch.float32).to(device)
            tgt = src.clone()
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

            output = cofs_model(src, layout, tgt, seq_num, tgt_num)

            # 计算损失
            loss = loss_calculate(src, output, config)

            # 清除梯度
            optimizer.zero_grad()

            #反向传播
            loss.backward()

            # 更新参数
            optimizer.step()
            scheduler.step()

            pbar.set_postfix(loss=loss)
            pbar.update(1)

        if epoch % checkpoint_freq == 0:
            # 保存训练参数
            torch.save(cofs_model.state_dict(), model_param_path + f'/bedrooms_model_{epoch}.pth')
            print(f"Model saved at Epoch: {epoch}")

    wandb.finish()

