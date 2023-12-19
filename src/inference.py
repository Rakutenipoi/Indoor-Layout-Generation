import sys
from process.dataset import *
from network.network import cofs_network
import time

# 系统设置
sys.setrecursionlimit(100000)
sys.stdout.flush = True

# pytorch设置
torch.set_printoptions(profile="full")
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

# GPU
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

# 读取config
config_path = 'config'
config_name = 'bedrooms_config.yaml'
config = read_file_in_path(config_path, config_name)

# 数据集部分参数
attr_num = config['data']['attributes_num']
class_num = config['data']['class_num']
permutation_num = config['data']['permutation_num']
max_sequence_length = config['network']['max_sequence_length']

# 训练数据读取
data_path = 'data/processed/bedrooms/'
layouts = np.load(os.path.join(data_path, 'bedrooms_simple_layout.npy'))

# 创建网络模型
cofs_model = cofs_network(config).to(device)

if __name__ == '__main__':
    # 设置为推理模式
    cofs_model.eval()

    for j in range(0, max_sequence_length, attr_num):







