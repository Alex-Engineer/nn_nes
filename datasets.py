"""
Datasets and dataloaders for moons; blobs; circles;

"""
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt

#todo add train/test dataloaders (if we don't know y for test dataset)
class Circles(Dataset):
    """
    Circles dataset
    """
    def __init__(self, n_samples, noise=None, shuffle=True, random_state=0, factor=0.8):
        self.X, self.y = datasets.make_circles(n_samples=n_samples, shuffle=shuffle, noise=noise,
                                               random_state=random_state,
                                               factor=factor)
        self.X , self.y =  self.X.astype(np.float32), self.y.astype(np.int)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))

    def plot_data(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=1)
        plt.show()


if __name__ == "__main__":
    circles_dataset = Circles(5000, noise=0.05)
    # print(circles_dataset.X)
    # print(len(circles_dataset))
    # print(circles_dataset[0])
    circles_dataset.plot_data()

    train_dataset = Circles(15, noise=0.05, random_state=0)
    test_dataset = Circles(1000, noise=0.05, random_state=1)

    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)
    # for x, y in train_dataloader:
    #     print(x, y)
