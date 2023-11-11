from torch.distributions.relaxed_bernoulli import LogitRelaxedBernoulli
import torch

class Logistic_Distribution:
    def __init__(self, means, variances):
        self.means = means
        self.variances = variances

    def log_prob(self, x):
        # 计算逻辑分布的对数概率密度函数
        log_prob = -torch.log(self.variances) - torch.abs((x - self.means) / self.variances) - 2 * torch.log(1 + torch.exp(-torch.abs((x - self.means) / self.variances)))

        return log_prob
