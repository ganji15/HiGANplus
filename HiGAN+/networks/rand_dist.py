import torch
import numpy as np
from copy import deepcopy

# Utility file to seed rngs
def seed_rng(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  np.random.seed(seed)


# A highly simplified convenience class for sampling from distributions
# One could also use PyTorch's inbuilt distributions package.
# Note that this class requires initialization to proceed as
# x = Distribution(torch.randn(size))
# x.init_distribution(dist_type, **dist_kwargs)
# x = x.to(device,dtype)
# This is partially based on https://discuss.pytorch.org/t/subclassing-torch-tensor/23754/2
class Distribution(torch.Tensor):
    # Init the params of the distribution
    def init_distribution(self, dist_type, **kwargs):
        seed_rng(kwargs['seed'])
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        if self.dist_type == 'normal':
            self.mean, self.var = kwargs['mean'], kwargs['var']
        elif self.dist_type == 'uniform':
            self.low, self.high = kwargs['low'], kwargs['high']
        elif self.dist_type == 'categorical':
            self.num_categories = kwargs['num_categories']
        elif self.dist_type == 'poisson':
            self.lam = kwargs['var']
        elif self.dist_type == 'gamma':
            self.scale = kwargs['var']

    def sample_(self):
        if self.dist_type == 'normal':
            self.normal_(self.mean, self.var)
        elif self.dist_type == 'uniform':
            self.uniform_(self.low, self.high)
        elif self.dist_type == 'categorical':
            self.random_(0, self.num_categories)
        elif self.dist_type == 'poisson':
            type = self.type()
            device = self.device
            data = np.random.poisson(self.lam, self.size())
            self.data = torch.from_numpy(data).type(type).to(device)
        elif self.dist_type == 'gamma':
            type = self.type()
            device = self.device
            data = np.random.gamma(shape=1, scale=self.scale, size=self.size())
            self.data = torch.from_numpy(data).type(type).to(device)
            # return self.variable
        return deepcopy(self).detach()

    # # Silly hack: overwrite the to() method to wrap the new object
    # # in a distribution as well
    def to(self, *args, **kwargs):
        new_obj = Distribution(self)
        new_obj.init_distribution(self.dist_type, **self.dist_kwargs)
        new_obj.data = super().to(*args, **kwargs)
        return new_obj

# Convenience function to prepare a z vector
def prepare_z_dist(G_batch_size, dim_z, device='cuda', seed=0):
    z_ = Distribution(torch.randn(G_batch_size, dim_z, requires_grad=False))
    z_.init_distribution('normal', mean=0, var=1.0, seed=seed)
    z_ = z_.to(device)
    return z_

# Convenience function to prepare a z vector
def prepare_y_dist(G_batch_size, nclasses, device='cuda', seed=0):
    y_ = Distribution(torch.zeros(G_batch_size, requires_grad=False))
    y_.init_distribution('categorical', num_categories=nclasses, seed=seed)
    y_ = y_.to(device, torch.int64)
    return y_
