import pickle
from torch.utils.data import DataLoader, SubsetRandomSampler
import tqdm
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import pandas as pd
from torch.cuda.amp import GradScaler, autocast

from network.network import cofs_network
from process.dataset import *
from utils.loss_cal import *
from utils.monitor import *

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(current_path)

    # pytorch设置
    np.set_printoptions(suppress=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_dtype(torch.float32)

    # GPU
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)

    # 读取config
    config_path = os.path.join(project_path, 'config')
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
    weight_decay = training_config['weight_decay']
    warmup_steps = training_config['warmup_steps']

    # wandb设置
    wandb_key = config['wandb']['key']

    # wandb设置
    wandb.login(key=wandb_key)
    wandb.init(
        # set the wandb project where this run will be logged
        project="cofs",

        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": "Transformer",
            "dataset": "3D-Front",
            "epochs": epochs,
            "dropout": dropout,
            "batch_size": batch_size,
            "max_sequence_length": max_sequence_length,
            "class_num": class_num,
            "encoder_layers": network_param['n_enc_layers'],
            "decoder_layers": network_param['n_dec_layers'],
            "heads": network_param['n_heads'],
            "dimensions": network_param['dimensions'],
            "feed_forward_dimensions": network_param['feed_forward_dimensions'],
            "activation": network_param['activation'],
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps
        }
    )

    # 训练数据读取
    data_path = os.path.join(project_path, 'data')
    dataset_name = 'dataset.pkl'
    dataset_path = os.path.join(data_path, dataset_name)
    has_dataset = os.path.exists(dataset_path)
    if has_dataset:
        with open(dataset_path, 'rb') as f:
            src_dataset = pickle.load(f)
        print("Dataset loaded")
    else:
        data_type = 'full_shuffled'
        full_data = np.load(os.path.join(data_path, f'bedrooms_{data_type}_data.npy'), allow_pickle=True)
        layouts = np.load(os.path.join(data_path, f'bedrooms_{data_type}_layout.npy'))
        sequences = np.load(os.path.join(data_path, f'bedrooms_{data_type}_sequence.npy'))
        # 提取数据
        sequence_index, layout_index, sequences_num = extract_data(full_data)
        # 数据集转换
        sequences = sequences.reshape(-1, max_sequence_length)
        src_dataset = COFSDataset(sequences, layouts, sequences_num)
        with open(dataset_path, 'wb') as f:
            pickle.dump(src_dataset, f)
        print("Dataset saved")

    # 对src_dataset进行划分
    train_size = int(0.8 * len(src_dataset))
    val_size = len(src_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(src_dataset, [train_size, val_size])
    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 创建网络模型
    cofs_model = cofs_network(config).to(device)

    # 从上次训练的模型参数开始训练
    load_pretrain = False
    model_param_path = os.path.join(project_path, 'model')
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


    def lr_lambda(lr_step):
        if lr_step < warmup_steps:
            return lr * lr_step / warmup_steps
        else:
            return lr

    # 优化器
    optimizer = torch.optim.AdamW(cofs_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler()

    # tqdm
    print("dataLoader length: ", len(train_dataLoader))
    pbar = tqdm.tqdm(total=epochs)

    # 设置为训练模式
    cofs_model.train()

    # 迭代训练
    for epoch in range(model_epoch_index + 1, epochs):
        avg_loss = 0
        epoch_time = time.time()
        for step, batch in enumerate(train_dataLoader):
            optimizer.zero_grad(set_to_none=True)
            # 读取数据
            src, layout, seq_num = batch
            seq_num = seq_num.to(torch.int32).to(device)
            tgt_num = seq_num
            # 读取序列
            src = src.to(device)
            # tgt = src.clone()
            # 读取布局
            layout = layout.to(device).to(torch.float32)
            # 设置requires_grad
            src.requires_grad = False
            layout.requires_grad = True
            # tgt.requires_grad = True

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

            with autocast():
                # 前向传播
                output = cofs_model(src, layout, src, seq_num, tgt_num)
                # 计算损失
                loss = loss_calculate(src, output, tgt_num, config)

            avg_loss += loss.item()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(parameters=cofs_model.parameters(), max_norm=30, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # 每隔5个epoch进行一次validation
        if epoch % 5 == 0:
            with torch.no_grad():
                cofs_model.eval()
                val_loss = 0
                for step, batch in enumerate(val_dataLoader):
                    # 读取数据
                    src, layout, seq_num = batch
                    seq_num = seq_num.to(torch.int32).to(device)
                    tgt_num = seq_num
                    # 读取序列
                    src = src.to(device)
                    # tgt = src.clone()
                    # 读取布局
                    layout = layout.to(device).to(torch.float32)
                    # 设置requires_grad
                    src.requires_grad = False
                    layout.requires_grad = False
                    # tgt.requires_grad = False
                    with autocast():
                        # 前向传播
                        output = cofs_model(src, layout, src, seq_num, tgt_num)
                        # 计算损失
                        loss = loss_calculate(src, output, tgt_num, config)
                    val_loss += loss.item()
                wandb.log({'val_loss': loss})
            cofs_model.train()

        # 更新pbar
        avg_loss /= len(train_dataLoader)
        pbar.set_postfix(avg_loss=avg_loss, time=time.time() - epoch_time)
        pbar.update(1)
        wandb.log({'train_loss': avg_loss})

        if epoch % checkpoint_freq == 0:
            # 保存训练参数
            if (not os.path.exists(model_param_path)):
                os.makedirs(model_param_path)
            torch.save(cofs_model.state_dict(), model_param_path + f'/bedrooms_model_{epoch}.pth')
            print(f"Model saved at Epoch: {epoch}")

    if (not os.path.exists(model_param_path)):
        os.makedirs(model_param_path)
    torch.save(cofs_model.state_dict(), model_param_path + f'/bedrooms_model_{epochs}.pth')
    wandb.finish()

