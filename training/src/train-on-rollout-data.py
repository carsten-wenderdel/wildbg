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
        # If you want to combine several CSV files from another folder, use the following: 
        # csv_files = [
        #     "../../wildbg-training/data/0006/rollouts.csv",
        #     "../../wildbg-training/data/0007/rollouts.csv",
        # ]
        csv_files = ["../training-data/rollouts.csv"]
        self.data = pd.concat([pd.read_csv(f, sep=';') for f in csv_files ], ignore_index=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # First 6 columns are outputs, last 202 columns are inputs
        output = self.data.iloc[idx, 0:6]
        input = self.data.iloc[idx, 6:208]
        return torch.FloatTensor(input).to(device), torch.FloatTensor(output).to(device)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(202, 300)
        self.hidden2 = nn.Linear(300, 250)
        self.hidden3 = nn.Linear(250, 200)

        # Output layer, 6 outputs for win/lose - normal/gammon/bg
        self.output = nn.Linear(200, 6)
        
        # Define activation function and softmax output 
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.activation(x)
        x = self.hidden2(x)
        x = self.activation(x)
        x = self.hidden3(x)
        x = self.activation(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

model = Network().to(device)

traindata = WildBgDataSet()
trainloader = DataLoader(traindata, batch_size=64, shuffle=True)

# Define loss function, L1Loss and MSELoss are good choices
criterion = nn.MSELoss()

# Optimizer based on model, adjust the learning rate
# 4.0 has worked well for Tanh(), one layer and 100k positions
# 3.0 has worked well for ReLu(), three layers and 200k positions
optimizer = torch.optim.SGD(model.parameters(), lr=3.0)

epochs = 20

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
