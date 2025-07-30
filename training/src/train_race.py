from pathlib import Path
import torch
from torch import nn, optim

from helper import train, Model


class TinyRaceModel(Model):
    def __init__(self):
        super().__init__()
        self.num_inputs = 186

        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(self.num_inputs, 16)
        # Output layer, 6 outputs for win/lose - normal/gammon/bg
        self.output = nn.Linear(16, 6)

        # Define activation function
        self.activation = nn.Tanh()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

    def optimizer(self):
        return optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)


class RaceModel(Model):

    def __init__(self):
        super().__init__()
        self.num_inputs = 186

        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(self.num_inputs, 300)
        self.hidden2 = nn.Linear(300, 250)
        self.hidden3 = nn.Linear(250, 200)

        # Output layer, 6 outputs for win/lose - normal/gammon/bg
        self.output = nn.Linear(200, 6)

        # Define activation function
        self.activation = nn.ReLU()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.hidden3(x)
        x = self.activation(x)
        x = self.output(x)
        return x

    def optimizer(self):
        return optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)


def main():
    # Make the training process deterministic
    torch.manual_seed(0)

    path = "./training-data/"
    Path(path).mkdir(exist_ok=True)

    # If you want to train a tiny model, change the following line:
    # model = TinyRaceModel()
    model = RaceModel()
    train(model, path + "race-inputs.csv", path + "race")

    print('Finished Training')


if __name__ == "__main__":
    main()
