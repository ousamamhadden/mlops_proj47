import os
import os.path as path

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.utils.data import DataLoader


class MyAwesomeModel(LightningModule):

    """My awesome model.

    This class defines a convolutional neural network model that can classify images into 10 classes. It consists of three convolutional layers, each followed by a max pooling layer, two fully connected layers, and a dropout and a batch normalization layer in between. The final layer uses a log softmax activation function to output the class probabilities.

    Attributes
    ----------
    conv1 : torch.nn.Conv2d
        The first convolutional layer with 1 input channel, 32 output channels, and a 3x3 kernel size with padding.
    conv2 : torch.nn.Conv2d
        The second convolutional layer with 32 input channels, 64 output channels, and a 3x3 kernel size with padding.
    conv3 : torch.nn.Conv2d
        The third convolutional layer with 64 input channels, 128 output channels, and a 3x3 kernel size with padding.
    pool : torch.nn.MaxPool2d
        The max pooling layer with a 2x2 kernel size and stride.
    fc1 : torch.nn.Linear
        The first fully connected layer with 128 * 3 * 3 input features and 256 output features.
    fc2 : torch.nn.Linear
        The second fully connected layer with 256 input features and 10 output features.
    dropout : torch.nn.Dropout
        The dropout layer with a probability of 0.25.
    bn : torch.nn.BatchNorm1d
        The batch normalization layer for the 256-dimensional feature vector.
    criterion : nn.NLLLoss
        The negative log likelihood loss function criterion.
    train_step_outputs : []
        Train steps list.
    train_losses : []
        Train losses list.
    test_step_outputs : []
        Test steps list.
    test_losses : []
        Test losses list.

    Methods
    -------
    forward(x)
        Defines the forward pass of the model, given an input tensor x.

    training_step(batch, batch_idx)
        Defines the training logic for a single batch, given the data and the batch index.

    on_train_epoch_end()
        Defines the logic for aggregating and logging the metrics at the end of each training epoch.

    test_step(batch, batch_idx)
        Defines the test logic for a single batch, given the data and the batch index.

    on_test_epoch_end()
        Defines the logic for aggregating and logging the metrics at the end of each test epoch.

    configure_optimizers()
        Returns a single optimizer, which is the stochastic gradient descent (SGD) with a learning rate of 0.01.

    train_dataloader()
        Returns a single dataloader, which loads the training set from a file and shuffles it with a batch size of 64.

    test_dataloader()
        Returns a single dataloader, which loads the test set from a file and shuffles it with a batch size of 64.
    """

    def __init__(self):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        # Batch normalization layer
        self.bn = nn.BatchNorm1d(256)

        self.criterion = nn.NLLLoss()

        self.train_step_outputs = []
        self.train_losses = []
        self.test_step_outputs = []
        self.test_losses = []
        # self.running_train_loss = 0

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
        # Convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the image
        x = x.view(-1, 128 * 3 * 3)
        # Dropout layer
        x = self.dropout(x)
        # Fully connected layer with batch normalization and ReLU activation
        x = F.relu(self.bn(self.fc1(x)))
        # Dropout layer
        x = self.dropout(x)
        # Fully connected layer with log softmax activation
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        output = {"loss": loss, "acc": acc}
        self.train_step_outputs.append(output)
        # self.running_train_loss += loss.item()
        # if batch_idx == self.train_loader_len - 1:
        #     train_loss = self.running_train_loss / self.train_loader_len
        #     print(f"Training loss: {train_loss}")
        #     self.train_losses.append(train_loss)
        #     self.running_train_loss = 0
        return output

    def on_train_epoch_end(self):
        # Access the outputs from the list
        outputs = self.train_step_outputs
        # Aggregate the metrics across the training batches
        train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        train_acc = torch.stack([x["acc"] for x in outputs]).mean()
        self.train_losses.append(train_loss.item())
        self.log("train_loss", train_loss)
        self.log("train_acc", train_acc)
        print(f"Training loss: {train_loss}")
        # Clear the list for the next epoch
        self.train_step_outputs = []

    def test_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterion(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        output = {"loss": loss, "acc": acc}
        self.test_step_outputs.append(output)
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        test_loss = torch.stack([x["loss"] for x in outputs]).mean()
        test_acc = torch.stack([x["acc"] for x in outputs]).mean()
        self.test_losses.append(test_loss.item())
        self.log("test_loss", test_loss)
        self.log("test_acc", test_acc)
        print(f"Test loss: {test_loss}")
        self.test_step_outputs = []

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)

    def train_dataloader(self):
        train_set = torch.load(path.join(os.getcwd(), "data/processed/train_set.pt"))
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        self.train_loader_len = len(train_loader)
        return train_loader

    def test_dataloader(self):
        test_set = torch.load(path.join(os.getcwd(), "data/processed/test_set.pt"))
        test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
        self.test_loader_len = len(test_loader)
        return test_loader
