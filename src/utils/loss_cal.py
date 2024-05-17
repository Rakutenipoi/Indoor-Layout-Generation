import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import wandb
import numpy as np

from .monitor import *

# GPU
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

def gaussian_loss(predict, truth):
    # 计算高斯分布的均值和方差
    mu = predict[:, :, :, 1]  # 预测结果的均值
    v = predict[:, :, :, 2]  # 预测结果的方差

    # 添加小量，防止除零
    epsilon = 1e-6
    v = torch.clamp(v, min=epsilon)

    # 计算logistic分布的概率密度函数值
    # gaussian_prob = torch.exp(-0.5 * torch.pow(truth.unsqueeze(-1) - mu, 2) / v) / (torch.sqrt(v * 2 * torch.pi))
    # diff_x = (truth.unsqueeze(-1) - mu) / v * 0.5
    # logistic_prob = 1 / (v * torch.pow(torch.exp(diff_x) + torch.exp(-diff_x), 2))
    normal_dist = dist.Normal(mu, torch.sqrt(v))
    gaussian_prob = normal_dist.log_prob(truth.unsqueeze(-1)).exp()

    # 计算每个高斯分布的权重
    weight = predict[:, :, :, 0]  # 预测结果的权重
    # weight = F.softmax(weight, dim=-1)

    # 将预测结果按照最后一个维度求和，得到所有高斯分布的混合概率密度函数值
    prob = gaussian_prob * weight
    mixed_gaussian_prob = torch.sum(prob, dim=-1)

    # 添加小量，防止取对数时出现NaN
    mixed_gaussian_prob = torch.clamp(mixed_gaussian_prob, min=epsilon)

    # 计算负对数似然损失函数值
    loss = -torch.log(mixed_gaussian_prob)

    return loss

def get_padding_mask(seq_length, batch_size=32, sequence_length=105):
    padding_mask = torch.ones((batch_size, sequence_length), device=device)
    for i, length in enumerate(seq_length):
        padding_mask[i, length.item():] = 0.0

    return padding_mask

def loss_calculate(src, output, src_len, config):
    # config
    attributes_num = config['data']['attributes_num']
    object_max_num = config['data']['object_max_num']
    class_num = config['data']['class_num'] + 2
    max_len = attributes_num * object_max_num
    sample_num = config['network']['sampler']['output_dimension'] // 3
    batch_size = src.size(0)

    # 提取分类结果
    output_class = output[:, ::attributes_num, :class_num]
    src_class = src[:, ::attributes_num]

    # 提取回归结果
    slices = [output[:, start:start + attributes_num - 1, :] for start in range(1, max_len, attributes_num)]
    output_attr = torch.cat(slices, dim=1)
    slices = [src[:, start:start + attributes_num - 1] for start in range(1, max_len, attributes_num)]
    src_attr = torch.cat(slices, dim=1)

    # 回归结果采样
    output_attr = output_attr.view(batch_size, -1, sample_num, 3)

    # 计算分类损失
    slices = [output_class[i, :src_len[i], :] for i in range(batch_size)]
    output_class = torch.cat(slices, dim=0)
    slices = [src_class[i, :src_len[i]] for i in range(batch_size)]
    src_class = torch.cat(slices, dim=0)
    loss_class = F.cross_entropy(output_class, src_class.to(torch.int64), reduction='sum') / batch_size

    # 计算回归损失
    mask = get_padding_mask(src_len * (attributes_num - 1), batch_size)
    loss_property = gaussian_loss(output_attr, src_attr)
    loss_property = loss_property * mask

    # 对损失进行平均
    loss_property = torch.sum(loss_property, dim=-1).mean()

    # 总损失
    loss = loss_class + loss_property

    return loss