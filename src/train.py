import pickle
import sys
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
from utils.atiss_loss import get_losses

isDebug = True if sys.gettrace() else False

if __name__ == '__main__':
    current_path = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(current_path)

    # pytorch设置
    np.set_printoptions(suppress=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_default_dtype(torch.float32)
    torch.set_printoptions(profile="full")

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
    decay_steps = training_config['decay_steps']
    decay_final_steps = training_config['decay_final_steps']
    final_lr = training_config['final_lr']
    random_mask = training_config['random_mask']

    # wandb设置
    wandb_enabled = config['wandb']['enable']
    wandb_key = config['wandb']['key']
    project_name = config['wandb']['project']
    architecture = config['wandb']['architecture']
    dataset_name = config['wandb']['dataset']

    # wandb设置
    if not isDebug and wandb_enabled:
        wandb_mode = 'online'
    else:
        wandb_mode = 'disabled'

    wandb.login(key=wandb_key)
    run = wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": architecture,
            "dataset": dataset_name,
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
        },
        mode=wandb_mode,
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
        data_name = 'dataset_1.pkl'
        with open(os.path.join(data_path, data_name), 'rb') as f:
            src_data = pickle.load(f)
        src_dataset = COFSDataset(src_data, config['data'])
        with open(dataset_path, 'wb') as f:
            pickle.dump(src_dataset, f)
        print("Dataset saved")

    # 对src_dataset进行划分
    train_size = int(0.8 * len(src_dataset))
    test_size = len(src_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(src_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 创建网络模型
    cofs_model = cofs_network(config).to(device)

    # 从上次训练的模型参数开始训练
    load_checkpoint = config['training']['load_checkpoint']
    model_param_path = os.path.join(project_path, 'model')
    if wandb_mode == 'online' and run.name is not None:
        model_param_path = os.path.join(model_param_path, run.name)
    if not os.path.exists(model_param_path):
        os.makedirs(model_param_path)
    model_epoch_index = 0
    # 读取该路径的文件
    if (load_checkpoint):
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
        elif lr_step > decay_final_steps:
            return final_lr
        elif lr_step < decay_steps:
            return lr
        else:
            return final_lr + (decay_final_steps - lr_step) / (decay_final_steps - decay_steps) * (lr - final_lr)

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
        avg_loss, avg_label_loss, avg_translation_loss, avg_size_loss, avg_angle_loss = 0, 0, 0, 0, 0
        epoch_time = time.time()
        for step, batch in enumerate(train_dataLoader):
            optimizer.zero_grad(set_to_none=True)
            # 读取数据
            batch = [b.to(device) for b in batch]
            layout, src, tgt, tgt_y, src_len = batch
            layout = layout.to(torch.float32)
            src_len = src_len * attr_num
            tgt_len = src_len - attr_num
            tgt_y_len = tgt_len
            # 设置requires_grad
            src.requires_grad = False
            layout.requires_grad = True
            tgt.requires_grad = True
            tgt_y.requires_grad = False

            # src随机mask
            if random_mask:
                mask = torch.ones_like(src)
                for i in range(batch_size):
                    num = src_len[i, 0]
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

            # 设置requires_grad为True
            src.requires_grad = True

            with autocast():
                # 前向传播
                output = cofs_model(src, layout, tgt, src_len, tgt_len)
                # 计算损失
                # loss = loss_calculate(tgt_y, output, tgt_len, config)
                label_loss, translation_loss, size_loss, angle_loss = get_losses(tgt_y, output, config, tgt_y_len)

            loss = label_loss + translation_loss + size_loss + angle_loss
            avg_loss += loss.item()
            avg_label_loss += label_loss.item()
            avg_translation_loss += translation_loss.item()
            avg_size_loss += size_loss.item()
            avg_angle_loss += angle_loss.item()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(parameters=cofs_model.parameters(), max_norm=30, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        # 每隔5个epoch进行一次validation
        if epoch % 5 == 0:
            with torch.no_grad():
                cofs_model.eval()
                val_loss, val_label_loss, val_translation_loss, val_size_loss, val_angle_loss = 0, 0, 0, 0, 0
                for step, batch in enumerate(val_dataLoader):
                    # 读取数据
                    # 读取数据
                    batch = [b.to(device) for b in batch]
                    layout, src, tgt, tgt_y, src_len = batch
                    layout = layout.to(torch.float32)
                    tgt_len = src_len - 1
                    tgt_y_len = tgt_len
                    # 设置requires_grad
                    src.requires_grad = False
                    layout.requires_grad = False
                    tgt.requires_grad = False
                    tgt_y.requires_grad = False
                    with autocast():
                        # 前向传播
                        output = cofs_model(src, layout, tgt, src_len, tgt_len)
                        # 计算损失
                        # loss = loss_calculate(tgt_y, output, tgt_len, config)
                        label_loss, translation_loss, size_loss, angle_loss = get_losses(tgt_y, output, config, tgt_y_len)
                    val_loss += loss.item()
                    val_label_loss += label_loss.item()
                    val_translation_loss += translation_loss.item()
                    val_size_loss += size_loss.item()
                    val_angle_loss += angle_loss.item()

                val_loss /= len(val_dataLoader)
                val_label_loss /= len(val_dataLoader)
                val_translation_loss /= len(val_dataLoader)
                val_size_loss /= len(val_dataLoader)
                val_angle_loss /= len(val_dataLoader)
                wandb.log(data={'val_loss': val_loss, 'val_label_loss': val_label_loss, 'val_translation_loss': val_translation_loss,
                                'val_size_loss': val_size_loss, 'val_angle_loss': val_angle_loss}, commit=False)

            cofs_model.train()

        # 更新pbar
        avg_loss /= len(train_dataLoader)
        avg_label_loss /= len(train_dataLoader)
        avg_translation_loss /= len(train_dataLoader)
        avg_size_loss /= len(train_dataLoader)
        avg_angle_loss /= len(train_dataLoader)
        pbar.set_postfix(avg_loss=avg_loss, time=time.time() - epoch_time)
        pbar.update(1)
        wandb.log(data={'train_loss': avg_loss, 'label_loss': avg_label_loss, 'translation_loss': avg_translation_loss,
                        'size_loss': avg_size_loss, 'angle_loss': avg_angle_loss})

        if epoch % checkpoint_freq == 0:
            # 保存训练参数
            torch.save(cofs_model, model_param_path + f'/bedrooms_model_{epoch}.pth')
            print(f"Model saved at Epoch: {epoch}")

    torch.save(cofs_model, model_param_path + f'/bedrooms_model_{epochs}.pth')
    wandb.finish()

