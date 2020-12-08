from torch import nn


class SimpleNN(nn.Module):
    "model creating MLP by passing it a list of layer sizes"

    def __init__(self, size_list):
        super().__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i], size_list[i + 1]))
            # todo add different activations
            layers.append(nn.ReLU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        layers.append(nn.Softmax())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    net = SimpleNN([2, 10, 8, 2])
    print(net)
