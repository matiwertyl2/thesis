import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

def sample_normal(mu, var, device='cuda'):
  dim1 = mu.size(0)
  dim2 = mu.size(1)
  sample = torch.randn(dim1, dim2, device=device) * torch.sqrt(var) + mu
  return sample.view(dim1, dim2, 1, 1)

class InfoVAE(nn.Module):
  def __init__(self, EQ, EHead, QHead, D, input_dim, c_dim, z_dim, device='cuda'):
    super().__init__()
    
    self.EQ = EQ
    self.EHead = EHead
    self.QHead = QHead
    self.D = D
    
    self.input_dim = input_dim
    self.c_dim = c_dim
    self.z_dim = z_dim
    self.device = device
        
    print("InfoVAE created correctly")

  def latent_representation(self, x, return_muvar=False):
    mu, var = self.EHead(self.EQ(x))
    return (sample_normal(mu, var), mu, var) if return_muvar else sample_normal(mu, var)
    
class VAE(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    
    self.encoder = encoder
    self.D = decoder
    
    print("VAE created correctly")

  def latent_representation(self, x, return_muvar=False):
    mu, var = self.encoder(x)
    return (sample_normal(mu, var), mu, var) if return_muvar else sample_normal(mu, var)