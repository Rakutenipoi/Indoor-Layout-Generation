import torch
import torch.nn as nn
import math

class PositionEncoding(nn.Module):
    def __init__(self, class_num, sequence_length, encode_levels=128, E_dims=256):
        super(PositionEncoding, self).__init__(class_num, encode_levels)
        self.E_class = nn.Parameter(torch.randn(class_num, E_dims))
        self.encode_levels = encode_levels
        self.sequence_length = sequence_length
        self.E_dims = E_dims

    def forward(self, x):
        sequence = torch.zeros(self.max_sequence_length, self.E_dims)
        idx = 0

        for token in x:
            if isinstance(token, int):
                sequence[idx] = self.E_class[x, :]
            else:
                angles = torch.arange(128).float()  # 生成角度向量，范围为0到127
                frequencies = 2 ** angles * math.pi  # 生成频率向量
                sin_encodings = torch.sin(frequencies * x.unsqueeze(1))  # 计算正弦编码
                cos_encodings = torch.cos(frequencies * x.unsqueeze(1))  # 计算余弦编码
                encoding = torch.stack([sin_encodings, cos_encodings], dim=2)  # 按照 sin、cos 排布的顺序进行拼接
                encoding = encoding.view(-1, 256)  # 将编码向量展平为一维向量
                sequence[idx] = encoding
            idx += 1

        return sequence

class RelativePositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, attributes_num=8, E_dims=256):
        super(RelativePositionEncoding, self).__init__()
        self.E_relative_position = nn.Parameter(torch.randn(attributes_num, E_dims))
        self.attributes_num = attributes_num
        self.E_dims = E_dims
        self.max_sequence_length = max_sequence_length

    def forward(self, x):
        sequence = torch.zeros(self.max_sequence_length, self.E_dims)
        idx = 0

        for token in x:
            sequence[idx] = self.E_relative_position[idx % self.attributes_num, :]
            idx += 1

        return sequence
class ObjectIndexEncoding(nn.Module):
    def __init__(self, max_sequence_length, object_max_num, attributes_num=8, E_dims=256):
        super().__init__()
        self.E_object_index = nn.Parameter(torch.randn(object_max_num, E_dims))
        self.E_dims = E_dims
        self.max_sequence_length = max_sequence_length
        self.attributes_num = attributes_num

    def forward(self, x):
        sequence = torch.zeros(self.max_sequence_length, self.E_dims)
        idx = 0

        for token in x:
            if idx % self.attributes_num == 0:
                type = token
            sequence[idx] = self.E_object_index[type, :]
            idx += 1

        return sequence

class AbsolutePositionEncoding(nn.Module):
    def __init__(self, max_sequence_length, object_num, E_dims=256):
        super().__init__()
        self.E_absolute_position = nn.Parameter(torch.randn(object_num, E_dims))
        self.E_dims = E_dims
        self.max_sequence_length = max_sequence_length

    def forward(self, x):
        sequence = torch.zeros(self.max_sequence_length, self.E_dims)
        idx = 0

        for token in x:
            sequence[idx] = self.E_absolute_position[idx / 8, :]
            idx += 1

        return sequence


