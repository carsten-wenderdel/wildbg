from torch import optim, nn


class Model(nn.Module):

    def __init__(self, num_inputs: int):
        super().__init__()

        self.num_inputs = num_inputs

        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(num_inputs, 300)
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

    def criterion(self):
        return nn.CrossEntropyLoss()

    def optimizer(self):
        return optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-2)


# Wrap a model with logits as output and add softmax so that all outputs add up to 1.
class Wrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.base_model(x)
        return self.softmax(logits)
