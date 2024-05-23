import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# GPU
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

def get_padding_mask(seq_len, batch_size=32, max_len=105):
    padding_mask = torch.ones((batch_size, max_len), device=device)
    for i, length in enumerate(seq_len):
        padding_mask[i, length.item():] = 0.0

    return padding_mask

def cross_entropy_loss(pred, target):
    """Cross entropy loss."""
    B, L, C = target.shape
    loss = torch.nn.functional.cross_entropy(
        pred.reshape(-1, C),
        target.reshape(-1, C).argmax(-1),
        reduction="none"
    ).reshape(B, L)

    return loss


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

    # B, L, C = target.shape
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

def extract_bbox_params_from_tensor(t):
    if isinstance(t, dict):
        class_labels = t["class_labels_tr"]
        translations = t["translations_tr"]
        sizes = t["sizes_tr"]
        angles = t["angles_tr"]
    else:
        assert len(t.shape) == 3
        class_labels = t[:, :, :-7]
        translations = t[:, :, -7:-4]
        sizes = t[:, :, -4:-1]
        angles = t[:, :, -1:]

    return class_labels, translations, sizes, angles

def _targets_from_tensor(X_target):
    # Make sure that everything has the correct shape
    # Extract the bbox_params for the target tensor
    target_bbox_params = extract_bbox_params_from_tensor(X_target)
    target = {}
    target["labels"] = target_bbox_params[0]
    target["translations_x"] = target_bbox_params[1][:, :, 0:1]
    target["translations_y"] = target_bbox_params[1][:, :, 1:2]
    target["translations_z"] = target_bbox_params[1][:, :, 2:3]
    target["sizes_x"] = target_bbox_params[2][:, :, 0:1]
    target["sizes_y"] = target_bbox_params[2][:, :, 1:2]
    target["sizes_z"] = target_bbox_params[2][:, :, 2:3]
    target["angles"] = target_bbox_params[3]

    return target

def get_losses(tgt_y, output, config, tgt_y_len):
    attributes_num = config['data']['attributes_num']
    class_num = config['data']['class_num'] + 3
    max_len = attributes_num * (config['data']['object_max_num'] + 2)

    # 提取分类结果
    output_class = output[:, ::attributes_num, :class_num]
    src_class = tgt_y[:, ::attributes_num]
    tgt_y = tgt_y.unsqueeze(-1)

    # 提取回归结果
    slice_trans_x = [output[:, start:start + 1, :] for start in range(1, max_len, attributes_num)]
    slice_trans_y = [output[:, start + 1:start + 2, :] for start in range(1, max_len, attributes_num)]
    slice_trans_z = [output[:, start + 2:start + 3, :] for start in range(1, max_len, attributes_num)]
    slice_sizes_x = [output[:, start + 3:start + 4, :] for start in range(1, max_len, attributes_num)]
    slice_sizes_y = [output[:, start + 4:start + 5, :] for start in range(1, max_len, attributes_num)]
    slice_sizes_z = [output[:, start + 5:start + 6, :] for start in range(1, max_len, attributes_num)]
    slice_angles = [output[:, start + 6:start + 7, :] for start in range(1, max_len, attributes_num)]
    translations_x = torch.cat(slice_trans_x, dim=1)
    translations_y = torch.cat(slice_trans_y, dim=1)
    translations_z = torch.cat(slice_trans_z, dim=1)
    sizes_x = torch.cat(slice_sizes_x, dim=1)
    sizes_y = torch.cat(slice_sizes_y, dim=1)
    sizes_z = torch.cat(slice_sizes_z, dim=1)
    angles = torch.cat(slice_angles, dim=1)
    slice_trans_x = [tgt_y[:, start:start + 1, :] for start in range(1, max_len, attributes_num)]
    slice_trans_y = [tgt_y[:, start + 1:start + 2, :] for start in range(1, max_len, attributes_num)]
    slice_trans_z = [tgt_y[:, start + 2:start + 3, :] for start in range(1, max_len, attributes_num)]
    slice_sizes_x = [tgt_y[:, start + 3:start + 4, :] for start in range(1, max_len, attributes_num)]
    slice_sizes_y = [tgt_y[:, start + 4:start + 5, :] for start in range(1, max_len, attributes_num)]
    slice_sizes_z = [tgt_y[:, start + 5:start + 6, :] for start in range(1, max_len, attributes_num)]
    slice_angles = [tgt_y[:, start + 6:start + 7, :] for start in range(1, max_len, attributes_num)]
    tgt_y_translations_x = torch.cat(slice_trans_x, dim=1)
    tgt_y_translations_y = torch.cat(slice_trans_y, dim=1)
    tgt_y_translations_z = torch.cat(slice_trans_z, dim=1)
    tgt_y_sizes_x = torch.cat(slice_sizes_x, dim=1)
    tgt_y_sizes_y = torch.cat(slice_sizes_y, dim=1)
    tgt_y_sizes_z = torch.cat(slice_sizes_z, dim=1)
    tgt_y_angles = torch.cat(slice_angles, dim=1)

    # For the class labels compute the cross entropy loss between the
    # target and the predicted labels
    # 将src_class转为one-hot编码
    src_class = torch.nn.functional.one_hot(src_class.to(torch.int64), num_classes=class_num)
    # label smoothing
    src_class = src_class * 0.9 + 0.1 / class_num
    label_loss = cross_entropy_loss(output_class, src_class)

    translation_loss = dmll(translations_x, tgt_y_translations_x)
    translation_loss += dmll(translations_y, tgt_y_translations_y)
    translation_loss += dmll(translations_z, tgt_y_translations_z)
    size_loss = dmll(sizes_x, tgt_y_sizes_x)
    size_loss += dmll(sizes_y, tgt_y_sizes_y)
    size_loss += dmll(sizes_z, tgt_y_sizes_z)
    angle_loss = dmll(angles, tgt_y_angles)

    property_loss = translation_loss + size_loss + angle_loss
    mask = get_padding_mask(seq_len=tgt_y_len, max_len=max_len // attributes_num)
    property_loss = property_loss * mask
    property_loss = torch.sum(property_loss, dim=-1).mean()
    label_loss = label_loss * mask
    label_loss = torch.sum(label_loss, dim=-1).mean()

    loss = label_loss + property_loss


    return loss

