import torch.nn as nn
import torch

class Sampler(nn.Module):
    def __init__(self, input_dimension, class_num, config):
        super().__init__()
        self.input_dimension = input_dimension
        self.hidden_input_dimension = config['hidden_input_dimension']
        self.hidden_output_dimension = config['hidden_output_dimension']
        self.output_dimension = config['output_dimension']
        self.class_num = class_num

        self.continue_layers = nn.Sequential(
            nn.Linear(self.input_dimension, self.hidden_input_dimension),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_input_dimension, self.hidden_output_dimension),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_output_dimension, self.output_dimension),
        )

        self.discrete_layer = nn.Sequential(
            nn.Linear(self.input_dimension, self.class_num),
            #nn.Softmax(dim=-1),
            #nn.ReLU(),
        )

    def forward(self, x, is_class):
        if is_class:
            x = self.discrete_layer(x)
        else:
            x = self.continue_layers(x)

        return x