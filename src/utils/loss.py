import torch.nn.functional as F
import torch.distributions as dist
import torch

from .math import Logistic_Distribution

# ground_truth(batch_size, 1)
def class_type_loss(predicted_class, ground_truth):
    batch_size = predicted_class.size(0)
    predict = predicted_class.squeeze()
    truth = ground_truth.squeeze().to(int)

    loss = F.cross_entropy(predict.float(), truth.float(), reduction='none')
    loss.requires_grad = True

    return loss

def property_loss_distribution(predicted_property, ground_truth):
    batch_size = predicted_property.size(0)
    predict = predicted_property.view(predicted_property.size(0), -1, 3)

    # 拆分混合权重、均值和方差
    weights = predict[:, :, 0]
    means = predict[:, :, 1]
    variances = predict[:, :, 2]
    # 创建分布对象
    logistics = Logistic_Distribution(means, variances)
    # 创建混合分布
    mix_logistics = dist.Categorical(weights.t())
    samples = mix_logistics.sample()

    # 对数概率密度
    log_probs = logistics.log_prob(samples)

    nll_loss = log_probs.mean()

    return nll_loss

def property_loss_single(predicted_property, ground_truth):
    predict = predicted_property.squeeze()
    truth = ground_truth.squeeze()

    loss = F.mse_loss(predict, truth)

    return loss