from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import Model
from tiny_model import TinyModel
from dataset import WildBgDataSet

def save_model(model: nn.Module, path: str, num_inputs: int) -> None:
    dummy_input = torch.randn(1, num_inputs, requires_grad=True, device=device)
    torch.onnx.export(model, dummy_input, path)


def train(model: nn.Module, trainloader: DataLoader, epochs: int) -> nn.Module:
    # L1Loss has had an advantage of 0.042 equity compared to MSELoss (both trained on 200k contact positions).
    criterion = nn.L1Loss()

    # Optimizer based on model, adjust the learning rate
    # 4.0 has worked well for SGD, MSELoss, Tanh(), one layer and 100k positions
    # 3.0 has worked well for SGD, MSELoss/L1Loss, ReLu(), three layers and 200k positions
    # 700e-6 == 0.0007 has worked well for Adam, L1Loss, ReLu(), three layers and 200k positions
    optimizer = torch.optim.Adam(model.parameters(), lr=290e-6)

    for epoch in range(epochs):
        epoch_loss = 0.0
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
            epoch_loss += loss.item()
        
        epoch_loss /= len(trainloader) / 64
        print(f'[Epoch: {epoch + 1}] loss: {epoch_loss:.5f}')
    
    return model

def main(model: nn.Module, data_path: str, model_path: str, num_inputs: int):
    traindata = WildBgDataSet(data_path)
    trainloader = DataLoader(traindata, batch_size=64, shuffle=True)

    try:
        model = train(model, trainloader, 50)
    finally:
        print('Finished Training')
        save_model(model, model_path, num_inputs)

if __name__ == "__main__":
    # "mps" takes more time than "cpu" on Macs, so let's ignore it for now.
    device = (
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    Path("../neural-nets").mkdir(exist_ok=True)

    mode = "race"
    match mode:
        case "contact":
            num_inputs = 202
            model = Model(num_inputs).to(device)
            main(model, "../training-data/contact-inputs.csv", "../neural-nets/contact.onnx", num_inputs)
        case "race":
            # `Race` has fewer inputs than `Contact`
            num_inputs = 186
            model = Model(num_inputs).to(device)
            main(model, "../training-data/race-inputs.csv", "../neural-nets/race.onnx", num_inputs)
        case "tiny_race":
            # This is used to be committed to the repository and not taking up much space.
            num_inputs = 186
            model = TinyModel(num_inputs).to(device)
            main(model, "../training-data/race-inputs.csv", "../neural-nets/race.onnx", num_inputs)
        case _:
            print("Invalid mode")
