import time

from torch.utils.data import DataLoader

from datasets import Circles
from net import SimpleNN
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NetTrainer():
    def __init__(self, model, criterion=None, optimizer=None):
        self.model = model

        # todo select loss
        self.criterion = nn.CrossEntropyLoss()
        # todo select optimizer
        self.optimizer = optim.Adam(model.parameters())

        cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda else "cpu")

    def fit(self, dataloader, n_epochs):
        start_time = time.time()
        running_loss = 0
        for i in range(n_epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                self.optimizer.zero_grad()  # .backward() accumulates gradients
                data = data.to(self.device)
                target = target.to(self.device)  # all data & model on same device

                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                running_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                # print("batch_idx: ", batch_idx, 'Training Loss: ', running_loss / (batch_idx + 1))

            end_time = time.time()
            running_loss /= len(dataloader)
            print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')
        print("End epoch: ", (i+1))



if __name__ == "__main__":
    # init NN
    net = SimpleNN([2, 10, 8, 2])
    print(net)

    # init datasets and dataloaders
    train_dataset = Circles(5000, noise=0.05, random_state=0)
    test_dataset = Circles(1000, noise=0.05, random_state=1)
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    trainer = NetTrainer(net)

    trainer.fit(train_dataloader, 10)


