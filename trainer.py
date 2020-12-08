import time
from torch.utils.data import DataLoader
from datasets import Circles
from net import SimpleNN
import torch
import torch.nn as nn
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

    # todo add validation after each z epochs
    def fit(self, train_dataloader, n_epochs):
        start_time = time.time()
        running_loss = 0
        for i in range(n_epochs):
            for batch_idx, (data, target) in enumerate(train_dataloader):
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
            running_loss /= len(train_dataloader)
            print('Training Loss: ', running_loss, 'Time: ', end_time - start_time, 's')
        print("End epoch: ", (i + 1))

    def predict(self, dataloader):
        all_outputs = torch.tensor([], device=self.device, dtype=torch.long)
        with torch.no_grad():
            self.model.eval()

            for batch_idx, (data, target) in enumerate(dataloader):
                print(batch_idx)
                data = data.to(self.device)
                target = target.to(self.device)
                outputs = self.model(data)

                _, predicted = torch.max(outputs.data, 1)
                print("predicted", predicted)
                all_outputs = torch.cat((all_outputs, predicted), 0)

        return all_outputs

    def predict_proba(self, dataloader):
        all_outputs = torch.tensor([], device=self.device)
        with torch.no_grad():
            self.model.eval()
            for batch_idx, (data, target) in enumerate(dataloader):
                print(batch_idx)
                data = data.to(self.device)
                target = target.to(self.device)
                outputs = self.model(data)
                all_outputs = torch.cat((all_outputs, outputs), 0)
        return all_outputs

    def calculate_metrics(self, y_true, y_predict, list_of_metrics):
        """
        рассчет метрик для предсказанных классов и для вероятностей
        :param y_true:
        :param y_predict:
        :param list_of_metrics:
        :return:
        """
        pass


if __name__ == "__main__":
    # init NN
    net = SimpleNN([2, 10, 8, 2])
    print(net)

    # init datasets and dataloaders
    train_dataset = Circles(5000, noise=0.05, random_state=0)
    test_dataset = Circles(1000, noise=0.05, random_state=1)
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    trainer = NetTrainer(net)

    trainer.fit(train_dataloader, 10)

    prediction = trainer.predict(test_dataloader)
    print(prediction)
    print(prediction.size())

    prediction_proba = trainer.predict_proba(test_dataloader)
    print(prediction_proba)
    print(prediction_proba.size())



