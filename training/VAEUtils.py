import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

def sample_normal(mu, var, device='cuda'):
  dim1 = mu.size(0)
  dim2 = mu.size(1)
  sample = torch.randn(dim1, dim2, device=device) * torch.sqrt(var) + mu
  return sample.view(dim1, dim2, 1, 1)

def kl_loss(mu, var):
  N = mu.size(0)
  return (-0.5 / N) * torch.sum(1.0 + (var + 1e-6).log() - mu.pow(2) - var)

def normal_nll_loss(x, mu, var):
    logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
    nll = -(logli.sum(1).mean())
    return nll
  
def information_loss(x, mu, var):
  return normal_nll_loss(x, mu, var)

def reconstruction_loss(reconstruction, x):
  N = reconstruction.size(0)
  
  return F.binary_cross_entropy(reconstruction, x, reduction='sum') / N