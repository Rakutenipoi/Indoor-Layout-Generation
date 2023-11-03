import torch
import torch.nn as nn
import math

class PositionEncoding(nn.Module):
    def __init__(self, class_num, encode_levels=128, E_dims=256):
        super(PositionEncoding, self).__init__(class_num, encode_levels)
        self.E_class = nn.Parameter(torch.randn(class_num, E_dims)) # E_class的存在形式未定
        self.encode_levels = encode_levels

    def forward(self, x):
        if isinstance(x, int):
            return self.E_class[x, :]
        else:
            angles = torch.arange(128).float()  # 生成角度向量，范围为0到127
            frequencies = 2 ** angles * math.pi  # 生成频率向量
            sin_encodings = torch.sin(frequencies * x.unsqueeze(1))  # 计算正弦编码
            cos_encodings = torch.cos(frequencies * x.unsqueeze(1))  # 计算余弦编码
            encodings = torch.stack([sin_encodings, cos_encodings], dim=2)  # 按照 sin、cos 排布的顺序进行拼接
            encodings = encodings.view(-1, 256)  # 将编码向量展平为一维向量
            return encodings
class RelativePositionEncoding(nn.Module):
    def __init__(self, attributes_num=8, E_dims=256):
        super(RelativePositionEncoding, self).__init__()
        self.E_relative_position = nn.Parameter(torch.randn(attributes_num, E_dims))

    def forward(self, x):
        return self.E_relative_position[x, :]
class ObjectIndexEncoding(nn.Module):
    def __init__(self, object_max_num, E_dims=256):
        super().__init__()
        self.E_object_index = nn.Parameter(torch.randn(object_max_num, E_dims))

    def forward(self, x):
        return self.E_object_index[x, :]

class AbsolutePositionEncoding(nn.Module):
    def __init__(self, object_num, E_dims=256):
        super().__init__()
        self.E_absolute_position = nn.Parameter(torch.randn(object_num, E_dims))

    def forward(self, x):
        return self.E_absolute_position[x, :]


