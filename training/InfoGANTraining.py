import time
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from .InfoGANUtils import *

def train_infogan(infogan, train_loader, num_epochs, params, device='cuda', plot_rec=False, plot_freq=1):
  
  # Loss for discrimination between real and fake images.
  criterionD = nn.BCELoss()
  # Loss for discrete latent code.
  criterionQ_dis = nn.CrossEntropyLoss()
  # Loss for continuous latent code.
  criterionQ_con = NormalNLLLoss()

  optimD = torch.optim.Adam([{'params': infogan.discriminator.parameters()}, {'params': infogan.dhead.parameters()}], 
                            lr=params['D_learning_rate'], 
                            betas=(params['beta1'], params['beta2']))
  optimG = torch.optim.Adam([{'params': infogan.generator.parameters()}, {'params' : infogan.qhead.parameters()}], 
                            lr=params['G_learning_rate'], 
                            betas=(params['beta1'], params['beta2']))
  
  # Fixed noise to see changes 
  z = torch.randn(100, infogan.num_z, 1, 1, device=infogan.device)
  fixed_noise = z
  if(infogan.num_disc_c != 0):
      idx = np.arange(infogan.disc_c_dim).repeat(10)
      dis_c = torch.zeros(100, infogan.num_disc_c, infogan.disc_c_dim, device=infogan.device)
      for i in range(infogan.num_disc_c):
          dis_c[torch.arange(0, 100), i, idx] = 1.0

      dis_c = dis_c.view(100, -1, 1, 1)

      fixed_noise = torch.cat((fixed_noise, dis_c), dim=1)

  if(infogan.num_con_c != 0):
      con_c = torch.rand(100, infogan.num_con_c, 1, 1, device=infogan.device) * 2 - 1
      fixed_noise = torch.cat((fixed_noise, con_c), dim=1)

  real_label = 1
  fake_label = 0

  # List variables to store results of training.
  img_list = []
  G_losses = []
  D_losses = []

  print("-"*25)
  print("Starting Training Loop...\n")
  print("-"*25)

  start_time = time.time()
  iters = 0

  infogan.train()

  for epoch in range(num_epochs):
      epoch_start_time = time.time()

      for i, data in enumerate(train_loader, 0):
          # Get batch size
          b_size = data.size(0)
          # Transfer data tensor to GPU/CPU (device)
          real_data = data.to(device)

          # Updating discriminator and DHead
          optimD.zero_grad()
          # Real data
          label = torch.full((b_size, ), real_label, device=device)
          output1 = infogan.discriminator(real_data)
          probs_real = infogan.dhead(output1).view(-1)
          loss_real = criterionD(probs_real, label)
          loss_real.backward()
          # Calculate gradients.

          # Fake data
          label.fill_(fake_label)
          noise, idx = noise_sample(infogan.num_disc_c, infogan.disc_c_dim, infogan.num_con_c, infogan.num_z, b_size, device)
          fake_data = infogan.generator(noise)
          output2 = infogan.discriminator(fake_data.detach())
          probs_fake = infogan.dhead(output2).view(-1)
          loss_fake = criterionD(probs_fake, label)
          loss_fake.backward()

          D_loss =  loss_real + loss_fake
          # Update parameters
          optimD.step()

          # Updating Generator and QHead
          optimG.zero_grad()

          # Fake data treated as real.
          output = infogan.discriminator(fake_data)
          probs_fake = infogan.dhead(output).view(-1)
          label.fill_(real_label)
          gen_loss = criterionD(probs_fake, label)

          # loss value for latent code
          q_logits, q_mu, q_var = infogan.qhead(output)
          dis_loss = 0
          target = torch.LongTensor(idx).to(device)
          for j in range(infogan.num_disc_c):
              dis_loss += criterionQ_dis(q_logits[:, j*infogan.disc_c_dim : (j+1)*infogan.disc_c_dim ], target[j])

          # Calculating loss for continuous latent code.
          con_loss = 0
          if (infogan.num_con_c != 0):
              ## con był mnożony przez 0.1 nie wiem czemu
              con_loss = criterionQ_con(noise[:, infogan.num_z+ infogan.num_disc_c * infogan.disc_c_dim : ].view(-1, infogan.num_con_c), q_mu, q_var) * 0.1
          # Calculate gradients.

          # Net loss for generator.
          G_loss = gen_loss + con_loss + dis_loss
          # Calculate gradients.
          G_loss.backward()
          # Update parameters.
          optimG.step()

          # Check progress of training.
          if i != 0 and i%100 == 0:
              print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                    % (epoch+1, num_epochs, i, len(train_loader), 
                      D_loss.item(), G_loss.item()))

          # Save the losses for plotting.
          G_losses.append(G_loss.item())
          D_losses.append(D_loss.item())

          iters += 1

      epoch_time = time.time() - epoch_start_time
      print("Time taken for Epoch %d: %.2fs" %(epoch + 1, epoch_time))
      
      # Generate image to check performance of generator.
      if(epoch % plot_freq == 0 and plot_rec):
          with torch.no_grad():
              gen_data = infogan.generator(fixed_noise).detach().cpu()
          plt.figure(figsize=(10, 10))
          plt.axis("off")
          plt.imshow(np.transpose(vutils.make_grid(gen_data, nrow=10, padding=2, normalize=True), (1,2,0)))
          plt.show()
          plt.close('all')