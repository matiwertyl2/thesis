import time
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from thesis.training.VAEUtils import *

def train_vae(model, train_loader, num_epochs, optimizer, params, device='cuda', plot_rec=False, plot_freq=1):
  model.to(device)
  model.train()
  
  for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    for i, data in enumerate(train_loader, 0):
      b_size = data.size(0)
      
      data = data.to(device)
      
      ###################################################################
      model.zero_grad()
      
      mu, var = model.encoder(data)
      latent_samples = sample_normal(mu, var)
      
      reconstruction = model.D(latent_samples)
      
      kulback_leiber = kl_loss(mu, var)
      rec_loss = reconstruction_loss(reconstruction, data)
      
      loss = rec_loss + 5 *kulback_leiber
      loss.backward()
      optimizer.step()
      
      ###################################################################
      
      if i != 0 and i%100 == 0:
         print('[%d/%d][%d/%d]\tLoss: %.4f\t KL: %.4f\t'
                % (epoch+1, num_epochs, i, len(train_loader), 
                        loss.item(), kulback_leiber.item()))
          
    epoch_time = time.time() - epoch_start_time
    print("epoch %d done in %.2fs" %(epoch+1, epoch_time))

    if(plot_rec and epoch % plot_freq == 0):
       with torch.no_grad():

        batch_x = next(iter(train_loader)).to(device)[:10]
        mu, var = model.encoder(batch_x)
        latent_repr = sample_normal(mu, var)
        reconstruction = model.D(latent_repr).detach().cpu()

        plt.figure(figsize=(10, 5))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(reconstruction, nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.show()
        plt.close('all')

        plt.figure(figsize=(10, 5))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(batch_x.detach().cpu(), nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.show()
        plt.close('all')
              
  
def train_infovae(model, train_loader, num_epochs, optimizer, params, device='cuda', plot_rec=False, plot_freq=1):

  model.to(device)
  model.train()

  for epoch in range(num_epochs):
    epoch_start_time = time.time()

    for i, data in enumerate(train_loader, 0):
      # Get batch size
      b_size = data.size(0)
      # Transfer data tensor to GPU/CPU (device)
      data = data.to(device)

      ##############################################################
      model.zero_grad()

      eq_res = model.EQ(data)
      mu, var = model.EHead(eq_res)
      latent_samples = sample_normal(mu, var)

      reconstruction = model.D(latent_samples)

      eq_res2 = model.EQ(reconstruction)
      mu_q, var_q = model.QHead(eq_res2)

      info_loss = information_loss(latent_samples[:, params['z_dim']:].squeeze(), mu_q, var_q)
      kulback_leiber = kl_loss(mu, var)
      rec_loss = reconstruction_loss(reconstruction, data)

      loss = rec_loss + 5 * kulback_leiber + 7 * info_loss
      loss.backward()
      optimizer.step()

      ###############################################################

      if i != 0 and i%100 == 0:
         print('[%d/%d][%d/%d]\tLoss: %.4f\t KL: %.4f\t Info: %.4f'
                % (epoch+1, num_epochs, i, len(train_loader), 
                        loss.item(), kulback_leiber.item(), info_loss.item()))


    epoch_time = time.time() - epoch_start_time
    print("epoch %d done in %.2fs" %(epoch+1, epoch_time))

    if(plot_rec and epoch % plot_freq == 0):
       with torch.no_grad():

        batch_x = next(iter(train_loader)).to(device)[:10]
        mu, var = model.EHead(model.EQ(batch_x))
        latent_repr = sample_normal(mu, var)
        reconstruction = model.D(latent_repr).detach().cpu()

        plt.figure(figsize=(10, 5))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(reconstruction, nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.show()
        plt.close('all')

        plt.figure(figsize=(10, 5))
        plt.axis("off")
        plt.imshow(np.transpose(vutils.make_grid(batch_x.detach().cpu(), nrow=10, padding=2, normalize=True), (1,2,0)))
        plt.show()
        plt.close('all')