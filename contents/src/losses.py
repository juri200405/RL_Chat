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
            m_shape = m.mean.shape[-1]
            self.target = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(m_shape, device=self.encoder_device), torch.eye(m_shape, device=self.encoder_device))
        closs_entropy_loss = self.label_loss_func(out, label)
        KL_loss = torch.distributions.kl.kl_divergence(m, self.target).sum()
        KL_weight = self.anneal_function(step)
        return (closs_entropy_loss + KL_weight * KL_loss) / self.batch_size, closs_entropy_loss, KL_loss, KL_weight

class MmdLoss:
    def __init__(self, loss_func, config):
        self.label_loss_func = loss_func
        self.n_latent = config.n_latent
        self.decoder_device = config.decoder_device
        self.batch_size = config.batch_size
        self.mmd_coefficient = config.mmd_coefficient

    def kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

    def forward(self, out, label, x):
        y = torch.randn(256, self.n_latent, device=self.decoder_device)
        xx_kernel = self.kernel(x, x)
        yy_kernel = self.kernel(y, y)
        xy_kernel = self.kernel(x, y)
        mmd = xx_kernel.mean() + yy_kernel.mean() - 2*xy_kernel.mean()
        closs_entropy_loss = self.label_loss_func(out, label)
        return closs_entropy_loss + (self.mmd_coefficient * mmd), closs_entropy_loss, mmd
