from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd

# "mps" takes more time than "cpu" on Macs, so let's ignore it for now.
device = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps"
    # if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class WildBgDataSet(Dataset):
    def __init__(self):
        self.data = pd.read_csv("../training-data/rollouts.csv", sep = ';')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # First 6 columns are outputs, last 202 columns are inputs
        output = self.data.iloc[idx, 0:6]
        input = self.data.iloc[idx, 6:208]
        return torch.FloatTensor(input).to(device), torch.FloatTensor(output).to(device)


class Network(nn.Module):
# Stolen from https://towardsdatascience.com/building-neural-network-using-pytorch-84f6e75f9a
# Number of input/output adapted to our use case
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(202, 150)
        # Output layer, 6 outputs for win/lose - normal/gammon/bg
        self.output = nn.Linear(150, 6)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

model = Network().to(device)

traindata = WildBgDataSet()
trainloader = DataLoader(traindata, batch_size=64, shuffle=True)

# Define loss function, L1Loss and MSELoss are good choices
criterion = nn.MSELoss()

# Optimizer based on model, adjust the learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 10

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        # set optimizer to zero grad to remove previous epoch gradients
        optimizer.zero_grad()
        # forward propagation
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # backward propagation
        loss.backward()
        # optimize
        optimizer.step()
        running_loss += loss.item()
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')

Path("../neural-nets").mkdir(exist_ok=True)
dummy_input = torch.randn(1, 202, requires_grad=True, device=device)
model_onnx = torch.onnx.export(model, dummy_input, "../neural-nets/wildbg.onnx")
