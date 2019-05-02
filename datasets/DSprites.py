from torch.utils.data import Dataset, DataLoader

## this works only if magic ! works (notebooks)
def download_data():
    ![ -e /content/dsprites-dataset ] || git clone https://github.com/deepmind/dsprites-dataset.git

class DspritesDataset(Dataset):
    self.file_path = 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'

    def __init__(self):
        super().__init__()
        download_data()
        self.data = np.load(self.file_path, encoding='latin1')
        
        self.imgs = self.data['imgs']
        self.latents_values = self.data['latents_values']
        self.latents_classes = self.data['latents_classes']
        self.metadata = self.data['metadata'][()]
        
        self.latents_sizes = self.metadata['latents_sizes']
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)


    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=size)

        return samples

    def __getitem__(self, index):
        image_grayscale = self.imgs[index,:,:].astype('float32')
        return torch.from_numpy(image_grayscale).unsqueeze(0).float()
     
    def __len__(self):
        return self.imgs.shape[0]