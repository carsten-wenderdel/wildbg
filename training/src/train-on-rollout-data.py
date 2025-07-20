from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import Model
from model import Wrapper
from tiny_model import TinyModel
from dataset import WildBgDataSet

def save_model(model: nn.Module, path: str, num_inputs: int) -> None:
    # add softmax so that the 6 outputs add up to exactly 1.0
    wrapper = Wrapper(model)
    dummy_input = torch.randn(1, num_inputs, requires_grad=True, device=device)
    torch.onnx.export(wrapper, dummy_input, path)


# `path_prefix` should be something like `../training-data/race-` or `../training-data/contact-`
# It will then be appended with the epoch number and `.onnx` extension.
def train(model: nn.Module, trainloader: DataLoader, path_prefix: str, epochs: int):
    # CrossEntropyLoss is the best for a multi-class classifier. The model has only logits as outputs,
    # we add softmax later when we save the model to the disk.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

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

        epoch_plus_one = epoch + 1
        print(f'[Epoch: {epoch_plus_one}] loss: {epoch_loss:.5f}')

        if epoch_plus_one > 4:
            # Save epochs for each iteration after the first couple of epochs have passed
            save_model(model, path_prefix + f"{epoch_plus_one:03}" + ".onnx", num_inputs)


def main(model: nn.Module, data_path: str, path_prefix: str, num_inputs: int):
    traindata = WildBgDataSet(data_path)
    trainloader = DataLoader(traindata, batch_size=64, shuffle=True)

    try:
        train(model, trainloader, path_prefix,120)
    finally:
        print('Finished Training')

if __name__ == "__main__":
    # Make the training process deterministic
    torch.manual_seed(0)

    # "mps" takes more time than "cpu" on Macs, so let's ignore it for now.
    device = (
        "cuda"
        if torch.cuda.is_available()
        # else "mps"
        # if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    path = "../training-data/"
    Path(path).mkdir(exist_ok=True)

    mode = "contact"
    match mode:
        case "contact":
            num_inputs = 202
            model = Model(num_inputs).to(device)
            main(model, path + "contact-inputs.csv", path + mode, num_inputs)
        case "race":
            # `Race` has fewer inputs than `Contact`
            num_inputs = 186
            model = Model(num_inputs).to(device)
            main(model, path + "race-inputs.csv", path + mode, num_inputs)
        case "tiny_race":
            # This is used to be committed to the repository and not taking up much space.
            num_inputs = 186
            model = TinyModel(num_inputs).to(device)
            main(model, path + "race-inputs.csv", path + mode, num_inputs)
        case _:
            print("Invalid mode")
