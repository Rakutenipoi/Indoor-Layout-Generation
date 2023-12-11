import math

import torch
import torch.nn as nn
from torch.autograd import Variable

# GPU
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

angles = torch.arange(128).double().to(torch.device(device))  # 生成角度向量，范围为0到127
frequencies = 2 ** angles * math.pi  # 生成频率向量

class Embedding(nn.Module):
    def __init__(self, class_num, sequence_length, encode_levels=128, E_dims=256):
        super(Embedding, self).__init__()
        self.E_class = nn.Parameter(torch.randn(class_num, E_dims))
        self.encode_levels = encode_levels
        self.sequence_length = sequence_length
        self.E_dims = E_dims

    def forward(self, x):
        batch_size = x.size(0)
        sequence_legnth = x.size(1)
        sequence = torch.zeros((batch_size, sequence_legnth, self.E_dims)).to(torch.device(device))

        for batch in range(batch_size):
            for token in range(sequence_legnth):
                value = x[batch, token]
                if token % 8 == 0:
                    class_type = int(value)
                    sequence[batch, token, :] = self.E_class[class_type, :]
                else:
                    sin_encodings = torch.sin(frequencies * value)  # 计算正弦编码
                    cos_encodings = torch.cos(frequencies * value)  # 计算余弦编码
                    encoding = torch.stack([sin_encodings, cos_encodings], dim=1)  # 按照 sin、cos 排布的顺序进行拼接
                    encoding = encoding.view(-1, 256).float()  # 将编码向量展平为一维向量
                    sequence[batch, token, :] = encoding

        return sequence

class RelativePositionEncoding(nn.Module):
    def __init__(self, attributes_num=8, E_dims=256):
        super(RelativePositionEncoding, self).__init__()
        self.E_relative_position = nn.Parameter(torch.randn(attributes_num, E_dims))
        self.attributes_num = attributes_num
        self.E_dims = E_dims

    def forward(self, x):
        batch_size = x.size(0)
        sequence_legnth = x.size(1)
        sequence = torch.zeros((batch_size, sequence_legnth, self.E_dims)).to(torch.device(device))

        for batch in range(batch_size):
            for token in range(sequence_legnth):
                sequence[batch, token] = self.E_relative_position[token % self.attributes_num, :]

        return sequence

class ObjectIndexEncoding(nn.Module):
    def __init__(self, object_max_num, attributes_num=8, E_dims=256):
        super().__init__()
        self.E_object_index = nn.Parameter(torch.randn(object_max_num, E_dims))
        self.E_dims = E_dims
        self.attributes_num = attributes_num

    def forward(self, x):
        batch_size = x.size(0)
        sequence_legnth = x.size(1)
        sequence = torch.zeros((batch_size, sequence_legnth, self.E_dims)).to(torch.device(device))

        for batch in range(batch_size):
            for token in range(sequence_legnth):
                # if token % 8 == 0:
                #     class_type = int(x[batch, token, :])
                sequence[batch, token] = self.E_object_index[int(token / self.attributes_num), :]

        return sequence

class AbsolutePositionEncoding(nn.Module):
    def __init__(self, object_num, attributes_num=8, E_dims=256):
        super().__init__()
        self.E_absolute_position = nn.Parameter(torch.randn(object_num, E_dims))
        self.E_dims = E_dims
        self.attributes_num = attributes_num

    def forward(self, x):
        batch_size = x.size(0)
        sequence_legnth = x.size(1)
        sequence = torch.zeros((batch_size, sequence_legnth, self.E_dims)).to(torch.device(device))

        for batch in range(batch_size):
            for token in range(sequence_legnth):
                sequence[batch, token] = self.E_absolute_position[int(token / self.attributes_num), :]

        return sequence


