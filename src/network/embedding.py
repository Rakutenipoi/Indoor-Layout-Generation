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

class Embedding(nn.Module):
    def __init__(self, class_num, sequence_length, encode_levels=128, E_dims=256, attribute_num=8, dropout=0.3):
        super(Embedding, self).__init__()
        self.E_class = nn.Parameter(torch.randn(class_num, E_dims))
        self.class_num = class_num
        self.encode_levels = encode_levels
        self.sequence_length = sequence_length
        self.E_dims = E_dims
        self.dropout = nn.Dropout(p=dropout)

        # embedding mask
        self.embedding_mask = torch.zeros(sequence_length, dtype=torch.uint8, device=device)
        self.embedding_mask[::attribute_num] = 1
        self.inverse_embedding_mask = torch.ones(sequence_length, dtype=torch.uint8, device=device)
        self.inverse_embedding_mask[::attribute_num] = 0

        # sinusoial encoding
        self.angles = torch.arange(self.E_dims / 2, device=device).double()  # 生成角度向量，范围为0到127
        self.frequencies = 2 ** self.angles * math.pi  # 生成频率向量
        self.frequencies = self.frequencies

    def float_embedding(self, x):
        x = x.unsqueeze(-1)
        sin_encodings = torch.sin(self.frequencies * x)  # 计算正弦编码
        cos_encodings = torch.cos(self.frequencies * x)  # 计算余弦编码
        float_embedding = torch.stack([sin_encodings, cos_encodings], dim=1)  # 按照 sin、cos 排布的顺序进行拼接

        return float_embedding.reshape(-1, self.E_dims).float()

    def class_embedding(self, x):
        out = self.E_class.index_select(0, x * self.embedding_mask.to(torch.int32))
        return out

    def get_embedding(self, int_token, batch_size):
        int_embedding = self.E_class.index_select(0, int_token)
        return self.dropout(int_embedding.expand(batch_size, 1, self.E_dims))

    def forward(self, x):
        batch_size = x.size(0)
        sequence_legnth = x.size(1)

        # 对整数类型的数据使用 Embedding
        int_embedding = [self.class_embedding(x[i, :].to(torch.int32)) * self.embedding_mask.unsqueeze(-1) for i in range(batch_size)]
        int_embedding = torch.stack(int_embedding, dim=0)

        # 对浮点类型的数据使用 Sinusoidal Encoding
        float_embedding = [self.float_embedding(x[i, :]) * self.inverse_embedding_mask.unsqueeze(-1) for i in range(batch_size)]
        float_embedding = torch.stack(float_embedding, dim=0)

        # 合并整数类型和浮点类型的嵌入向量
        embedding = int_embedding + float_embedding

        return self.dropout(embedding)

class RelativePositionEncoding(nn.Module):
    def __init__(self, attributes_num=8, E_dims=256, object_max_num=15):
        super(RelativePositionEncoding, self).__init__()
        self.E_relative_position = nn.Parameter(torch.randn(attributes_num, E_dims, device=device))
        self.E_property_relative_position = nn.Parameter(torch.randn(4, E_dims, device=device))
        self.relative_index = torch.arange(attributes_num, device=device, dtype=torch.int32).repeat(object_max_num)
        self.property_relative_index = torch.tensor([0, 1, 1, 1, 2, 2, 2, 3], dtype=torch.int32, device=device).repeat(object_max_num)
        self.attributes_num = attributes_num
        self.E_dims = E_dims

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        out = self.E_relative_position.index_select(0, self.relative_index)
        out = out.expand(batch_size, seq_length, self.E_dims)
        property_out = self.E_property_relative_position.index_select(0, self.property_relative_index)
        property_out = property_out.expand(batch_size, seq_length, self.E_dims)

        return out + property_out

class ObjectIndexEncoding(nn.Module):
    def __init__(self, object_max_num, attributes_num=8, E_dims=256):
        super().__init__()
        self.E_object_index = nn.Parameter(torch.randn(object_max_num, E_dims))
        self.relative_index = torch.arange(object_max_num * attributes_num, device=device, dtype=torch.int32) // attributes_num
        self.E_dims = E_dims
        self.attributes_num = attributes_num

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        out = self.E_object_index.index_select(0, self.relative_index)
        out = out.expand(batch_size, seq_length, self.E_dims)

        return out

class AbsolutePositionEncoding(nn.Module):
    def __init__(self, object_max_num, attributes_num=8, E_dims=256):
        super().__init__()
        self.E_absolute_position = nn.Parameter(torch.randn(object_max_num * attributes_num, E_dims))
        self.relative_index = torch.arange(object_max_num * attributes_num, device=device, dtype=torch.int32)
        self.E_dims = E_dims
        self.attributes_num = attributes_num

    def forward(self, x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        out = self.E_absolute_position.index_select(0, self.relative_index)
        out = out.expand(batch_size, seq_length, self.E_dims)

        return out


