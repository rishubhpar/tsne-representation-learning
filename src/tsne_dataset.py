import torch
from torch.utils.data import Dataset, DataLoader


class tsne_dataset(Dataset):
    def __init__(self, dataset, tsne_embeddings):    
        self.dataset = dataset
        self.embds = tsne_embeddings
        print("TSNE dataset - Dataset shape: {}, Embds shape: {}".format(len(self.dataset), self.embds.shape))

    def __len__(self):
        return self.embds.shape[0]

    def __getitem__(self, idx):
        x = self.dataset[idx][0]
        y = self.dataset[idx][1]
        embds = torch.Tensor(self.embds[idx])

        # print("x shape: ", x.shape, "y", y, "embds shape: ", embds.shape)
        return x,embds,y
    
