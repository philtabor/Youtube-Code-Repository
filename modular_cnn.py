import torch as T
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

class CNNCell(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNNCell, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels,
                              kernel_size=3,
                              out_channels=output_channels)
        self.bn = nn.BatchNorm2d(num_features=output_channels)
        self.relu = nn.ReLU()

    def forward(self, batch_data):
        output = self.conv(batch_data)
        output = self.bn(output)
        output = self.relu(output)

        return output

class CNNNetwork(nn.Module):
    def __init__(self, lr, batch_size, n_classes, epochs):
        super(CNNNetwork,  self).__init__()
        self.lr = lr
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.epochs = epochs
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.loss_history = []
        self.acc_history = []


        self.cell1 = CNNCell(input_channels=1, output_channels=32)
        self.cell2 = CNNCell(input_channels=32, output_channels=32)
        self.cell3 = CNNCell(input_channels=32, output_channels=32)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2)

        self.cell4 = CNNCell(input_channels=32, output_channels=64)
        self.cell5 = CNNCell(input_channels=64, output_channels=64)
        self.cell6 = CNNCell(input_channels=64, output_channels=64)

        self.max_pool2 = nn.MaxPool2d(kernel_size=2)

        self.network = nn.Sequential(self.cell1, self.cell2, self.cell3,
                        self.max_pool1, self.cell4, self.cell5, self.cell6,
                        self.max_pool2)

        self.fc = nn.Linear(in_features=256, out_features=n_classes)
        self.loss = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.to(self.device)
        self.get_data()

    def forward(self, batch_data):
        batch_data = T.tensor(batch_data).to(self.device)
        output = self.network(batch_data)
        output = output.view(-1, 256)
        output = self.fc(output)

        return output

    def get_data(self):
        mnist_train_data = MNIST('mnist/', train=True,
                                 download=True, transform=ToTensor())

        self.train_data_loader = T.utils.data.DataLoader(mnist_train_data,
                        batch_size=self.batch_size, shuffle=True, num_workers=8)

        mnist_test_data = MNIST('mnist/', train=False,
                                 download=True, transform=ToTensor())

        self.test_data_loader = T.utils.data.DataLoader(mnist_test_data,
                        batch_size=self.batch_size, shuffle=True, num_workers=8)

    def _train(self):
        self.train()
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            for j, (input, label) in enumerate(self.train_data_loader):
                self.optimizer.zero_grad()
                label = label.to(self.device)
                prediction = self.forward(input)
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(classes != label,
                                T.tensor([1.]).to(self.device),
                                T.tensor([0.]).to(self.device))
                acc = 1 - T.sum(wrong) / self.batch_size
                loss = self.loss(prediction, label)
                self.acc_history.append(acc.item())
                ep_loss += loss.item()
                ep_acc.append(acc.item())
                loss.backward()
                self.optimizer.step()
            print('Finish epoch ', i, 'total loss %.3f training accuracy %.3f' % \
                                    (ep_loss, np.mean(ep_acc)))
            self.loss_history.append(ep_loss)

    def _test(self):
        self.eval()

        ep_loss = 0
        ep_acc = []
        for j, (input, label) in enumerate(self.test_data_loader):
            label = label.to(self.device)
            prediction = self.forward(input)
            classes = T.argmax(prediction, dim=1)
            wrong = T.where(classes != label,
                            T.tensor([1.]).to(self.device),
                            T.tensor([0.]).to(self.device))
            acc = 1 - T.sum(wrong) / self.batch_size
            loss = self.loss(prediction, label)
            ep_acc.append(acc.item())
            ep_loss += loss.item()
        print('Total loss %.3f accuracy %.3f' % (ep_loss, np.mean(ep_acc)))

if __name__ == '__main__':
    network = CNNNetwork(lr=0.001, batch_size=32, epochs=10, n_classes=10)
    network._train()
    plt.plot(network.loss_history)
    plt.show()
    plt.plot(network.acc_history)
    plt.show()
    network._test()
