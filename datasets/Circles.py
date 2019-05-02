import numpy as np 
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# synthetic dataset that is generated based on random generative factors.
# Each data sample can be defined by 6 variables, given in the order 
# x, y - uniform(0.1, 0.9) : position of circle
# r - uniform(0.05, 0.4) : radius of the circle
# r, g, b - uniform(0, 1) : color of the circle 
class GrayCirclesDataset(Dataset):
    img_size = 64
    z = 3
    
    def __init__(self, n):
        self.n = n 
        self.random_factors = self.__generate_random_factors()
        imgs = []
        for i in range(self.n):
            img = self.__generate_blob(self.random_factors[i, :])
            imgs.append(img.unsqueeze(0))

        self.data = torch.cat(imgs, 0)


    def __generate_blob(self, random_factors):
        x = random_factors[0]
        y = random_factors[1]
        r = random_factors[2]

        image = Image.new('L', (self.img_size, self.img_size))
        draw = ImageDraw.Draw(image)

        lx = (x - r) * self.img_size
        rx = (x + r) * self.img_size
        ly = (y - r) * self.img_size
        ry = (y + r) * self.img_size

        draw.ellipse((lx, ly, rx, ry), fill = 128)
        trans = transforms.ToTensor()
        return trans(image)

    def __generate_random_factors(self):
        position = torch.empty(self.n, 2).uniform_(0.1, 0.9)
        radius = torch.empty(self.n, 1).uniform_(0.05, 0.4)
        return torch.cat((position, radius), 1)
      
    def __getitem__(self, index):
        return self.data[index]
     
     
    def __len__(self):
        return self.data.size(0)


# synthetic dataset that is generated based on random generative factors.
# Each data sample can be defined by 6 variables, given in the order 
# x, y - uniform(0.1, 0.9) : position of circle
# r - uniform(0.05, 0.4) : radius of the circle
# r, g, b - uniform(0, 1) : color of the circle 
class ColorCirclesDataset(Dataset):
    img_size = 64
    z = 6
    
    def __init__(self, n):
        self.n = n 
        self.random_factors = self.__generate_random_factors()
        imgs = []
        for i in range(self.n):
            img = self.__generate_blob(self.random_factors[i, :])
            imgs.append(img.unsqueeze(0))

        self.data = torch.cat(imgs, 0)


    def __generate_blob(self, random_factors):
        x = random_factors[0]
        y = random_factors[1]
        r = random_factors[2]
        color = random_factors[3:] * 255

        image = Image.new('RGB', (self.img_size, self.img_size))
        draw = ImageDraw.Draw(image)

        lx = (x - r) * self.img_size
        rx = (x + r) * self.img_size
        ly = (y - r) * self.img_size
        ry = (y + r) * self.img_size

        draw.ellipse((lx, ly, rx, ry), fill = tuple(color))
        trans = transforms.ToTensor()
        return trans(image)

    def __generate_random_factors(self):
        position = torch.empty(self.n, 2).uniform_(0.1, 0.9)
        radius = torch.empty(self.n, 1).uniform_(0.05, 0.4)
        color = torch.empty(self.n, 3).uniform_(0, 1) 
        return torch.cat((position, radius, color), 1)
      
    def __getitem__(self, index):
        return self.data[index]
     
     
    def __len__(self):
        return self.data.size(0)
