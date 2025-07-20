from torch import nn

class TinyModel(nn.Module):
    def __init__(self, num_inputs: int):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(num_inputs, 16)
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
