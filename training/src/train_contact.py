from pathlib import Path
import torch
from torch import nn, optim
from helper import train, Model


class ContactModel(Model):

    def __init__(self):
        super().__init__()
        self.num_inputs = 202

        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(self.num_inputs, 300)
        self.hidden2 = nn.Linear(300, 250)
        self.hidden3 = nn.Linear(250, 200)

        # Output layer, 6 outputs for win/lose - normal/gammon/bg
        self.output = nn.Linear(200, 6)

        # Define activation function
        self.activation = nn.Hardsigmoid()

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
        return optim.AdamW(self.parameters(), lr=1130e-6)


def main():
    # Make the training process deterministic
    torch.manual_seed(0)

    path = "./training-data/"
    Path(path).mkdir(exist_ok=True)

    model = ContactModel()
    train(model, path + "contact-inputs.csv", path + "contact")

    print('Finished Training')


if __name__ == "__main__":
    main()
