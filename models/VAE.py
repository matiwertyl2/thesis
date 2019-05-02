import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

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
    
class VAE(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    
    self.encoder = encoder
    self.D = decoder
    
    print("VAE created correctly")