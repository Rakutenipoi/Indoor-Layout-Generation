import torch
from torchviz import make_dot

from network.network import cofs_network
from utils.yaml_reader import read_yaml

# GPU
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# 读取config
config = read_yaml('config/bedrooms_config.yaml')

# 训练数据设置


# 训练数据读取
layout_image = torch.randn(1, 1, 256, 256)
src_data = []
tgt_data = []
num_elements = 200
for i in range(num_elements):
    if i % 8 == 0 or (i-1) % 8 == 0:
        src_data.append(int(i / 8))
        tgt_data.append(int(i / 8))
    else:
        src_data.append(float(i / 8))
        tgt_data.append(float(i / 8))

src_data = torch.tensor(src_data)
tgt_data = torch.tensor(tgt_data)

#src_data = torch.unsqueeze(src_data, 0)
#tgt_data = torch.unsqueeze(tgt_data, 0)

# 创建网络模型
transformer = cofs_network(config)
#print(transformer)


if __name__ == '__main__':
    output = transformer(src_data, layout_image, tgt_data)
    make_dot(output, params=dict(list(transformer.named_parameters()))).render("transformer_torchviz", format="png")




    print('done')