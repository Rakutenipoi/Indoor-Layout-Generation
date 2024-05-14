import torch.nn as nn
import torch

# GPU
if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
device = torch.device(dev)

class Sampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dimension = config['network']['dimensions']
        self.class_num = config['network']['sampler']['output_dimension']
        self.hidden_input_dimension = config['network']['sampler']['hidden_input_dimension']
        self.hidden_output_dimension = config['network']['sampler']['hidden_output_dimension']
        self.output_dimension = config['network']['sampler']['output_dimension']
        self.attributes_num = config['data']['attributes_num']
        self.object_max_num = config['data']['object_max_num']
        self.max_len = self.attributes_num * self.object_max_num

        self.continue_layers = nn.Sequential(
            nn.Linear(self.input_dimension, self.hidden_input_dimension),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_input_dimension, self.hidden_output_dimension),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_output_dimension, self.output_dimension),
            nn.ReLU(),
        )

        self.discrete_layer = nn.Sequential(
            nn.Linear(self.input_dimension, self.class_num),
            #nn.Softmax(dim=-1),
            #nn.ReLU(),
        )

        mask = torch.zeros(self.max_len, dtype=torch.float32)
        mask[::self.attributes_num] = 1
        mask = mask.unsqueeze(0).expand(self.max_len, -1)
        self.mask = mask.to(device)
        inv_mask = torch.ones(self.max_len, dtype=torch.float32)
        inv_mask[::self.attributes_num] = 0
        inv_mask = inv_mask.unsqueeze(0).expand(self.max_len, -1)
        self.inv_mask = inv_mask.to(device)

    def forward(self, x):
        discrete_layer = self.discrete_layer(x)
        discrete_layer = discrete_layer.reshape(-1, self.max_len, self.class_num)
        discrete_layer = torch.matmul(self.mask, discrete_layer)

        continue_layer = self.continue_layers(x)
        continue_layer = continue_layer.reshape(-1, self.max_len, self.output_dimension)
        continue_layer = torch.matmul(self.inv_mask, continue_layer)

        x = discrete_layer + continue_layer

        return x