import decimal
import sys

import torch

class VaeLoss:
    def __init__(self, loss_func, config, len_itr):
        self.x0 = config.x0_epoch * len_itr
        self.k = config.anneal_k
        self.encoder_device = config.encoder_device
        self.batch_size = config.batch_size

        self.label_loss_func = loss_func
        self.target = None

    def anneal_function(self, step):
        tmp = 1/(1+decimal.Decimal(-self.k*(step-self.x0)).exp())
        if tmp < sys.float_info.min:
            tmp = sys.float_info.min
        return float(tmp)

    def forward(self, out, label, m, step):
        if self.target is None:
            print("calc target")
            m_shape = m.mean.shape[-1]
            self.target = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(m_shape, device=self.encoder_device), torch.eye(m_shape, device=self.encoder_device))
        closs_entropy_loss = self.label_loss_func(out, label)
        KL_loss = torch.distributions.kl.kl_divergence(m, self.target).sum()
        KL_weight = self.anneal_function(step)
        return (closs_entropy_loss + KL_weight * KL_loss) / self.batch_size, closs_entropy_loss, KL_loss, KL_weight
