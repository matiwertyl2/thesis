import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import Dataset, DataLoader

# Directory containing the data.
root = '/'

class MNISTDataset(Dataset):
  def __init__(self):
    super().__init__()
    transform = transforms.Compose([
      transforms.Resize(28),
      transforms.CenterCrop(28),
      transforms.ToTensor()])
    
    self.dataset = dsets.MNIST(root+'mnist/', train='train', 
                                download=True, transform=transform)
  
  def __getitem__(self, index):
    return self.dataset.data[index, :, :].unsqueeze(0).float() / 255
    
  def __len__(self):
    return self.dataset.data.shape[0]