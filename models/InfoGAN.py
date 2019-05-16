import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoGAN(nn.Module):
  def __init__(self, discriminator, qhead, dhead, generator, input_dim, num_z, num_disc_c, disc_c_dim, num_con_c, device='cuda'):
    super().__init__()
    
    self.discriminator = discriminator.to(device)
    self.qhead = qhead.to(device)
    self.dhead = dhead.to(device)
    self.generator = generator.to(device)
    
    self.input_dim = input_dim
    self.num_z = num_z
    self.num_disc_c = num_disc_c
    self.disc_c_dim = disc_c_dim
    self.num_con_c = num_con_c
    
    self.device = device
    
    self.validate_net_dims()
    
    print("InfoGAN model created, seems that dimensions are correct ;)")
   
  def validate_net_dims(self):
    batch_size = 13
    x = torch.zeros((batch_size,) + self.input_dim).to(self.device)
    
    
    x = self.discriminator(x)
    dhead_out = self.dhead(x)
    disc_logits, mu, var = self.qhead(x)
    
    if dhead_out.size() != torch.Size((batch_size, 1, 1, 1)):
      raise RuntimeError("DHead dim wrong, expected {0}, got {1}".format(str((batch_size, 1, 1, 1)), dhead_out.size()))
      
    if self.num_disc_c > 0:
      if disc_logits.size() != torch.Size((batch_size, self.num_disc_c * self.disc_c_dim)):
        raise RuntimeError("Disc logits dim wrong, expected {0}, got {1}".format((batch_size, self.num_disc_c * self.disc_c_dim), disc_logits.size()))
    
    if self.num_con_c > 0:
      if mu.size() != torch.Size((batch_size, self.num_con_c)):
        raise RuntimeError("Mu dim wrong, expected {0}, got {1}".format((batch_size, self.num_con_c), mu.size()))

      if var.size() != torch.Size((batch_size, self.num_con_c)):
        raise RuntimeError("Var dim wrong, expected {0}, got {1}".format((batch_size, self.num_con_c), var.size()))
      
    gen_input = torch.zeros((batch_size,self.num_con_c + self.num_z +  self.disc_c_dim * self.num_disc_c, 1, 1)).to(self.device)
    
    output = self.generator(gen_input)
    
    if output.size() != torch.Size((batch_size, ) + self.input_dim):
      raise RuntimeError("Generator output dim wrong, expected {0}, got {1}}".format((batch_size, ) + self.input_dim, output.size()))