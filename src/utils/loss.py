import torch.nn.functional as F
import torch.distributions as dist
import torch
import torch.nn as nn
import numpy as np

from .math import Logistic_Distribution

# GPU
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

# ground_truth(batch_size, 1)
def class_type_loss(predicted_class, ground_truth, class_num):
    truth = ground_truth.to(int)

    # one-hot编码
    output_type_one_hot = torch.zeros(predicted_class.size(0), class_num).to(device)
    src_data_class_one_hot = torch.zeros(ground_truth.size(0), class_num).to(device)
    output_type_one_hot.scatter_(1, predicted_class, 1)
    src_data_class_one_hot.scatter_(1, truth, 1)

    loss = torch.nn.CrossEntropyLoss(src_data_class_one_hot, truth)
    loss.requires_grad = True

    return loss

def class_type_loss_cross_nll(predict, truth):
    truth = truth.squeeze().to(int)
    m = nn.LogSoftmax(dim=1)
    nll_loss = nn.NLLLoss()

    output = nll_loss(m(predict), truth)

    return output

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

    mse_loss = nn.MSELoss()
    output = mse_loss(predict, truth)

    return output

def log_sum_exp(x):
    """Numerically stable log_sum_exp implementation that prevents
    overflow.
    """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))

def dmll(pred, target, log_scale_min=-7.0, num_classes=256):
    """Discretized mixture of logistic distributions loss
    Note that it is assumed that input is scaled to [-1, 1].

    Code adapted
    from https://github.com/idiap/linear-transformer-experiments/blob/0a540938ec95e1ec5b159ceabe0463d748ba626c/image-generation/utils.py#L31

    Arguments
    ----------
        pred (Tensor): Predicted output (B x L x T)
        target (Tensor): Target (B x L x 1).
        log_scale_min (float): Log scale minimum value
        num_classes (int): Number of classes

    Returns:
    --------
        Tensor: loss
    """

    B, L, C = target.shape
    nr_mix = pred.shape[-1] // 3

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = pred[:, :, :nr_mix]
    means = pred[:, :, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(
        pred[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min
    )

    centered_y = target - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(torch.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - torch.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()

    inner_inner_out = inner_inner_cond * \
        torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
        (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (target > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + \
        (1. - inner_cond) * inner_inner_out
    cond = (target < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)
    return -log_sum_exp(log_probs)










    