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
    dummy_input = torch.randn(1, num_inputs, requires_grad=True)
    torch.onnx.export(wrapper, dummy_input, path)


# `path_prefix` should be something like `./training-data/race-` or `./training-data/contact-`
# It will then be appended with the epoch number and `.onnx` extension.
def train(model: Model, data_path: str, path_prefix: str):
    # Import rollout data
    rollout_data = WildBgDataSet(data_path)
    train_loader = DataLoader(rollout_data, batch_size=64, shuffle=True)

    # "mps" takes more time than "cpu" on Macs, so let's ignore it for now.
    device = "cpu"
    print(f"Using {device} device")

    # CrossEntropyLoss is the best for a multi-class classifier. The model has only logits as outputs,
    # we add softmax later when we save the model to the disk.
    criterion = model.criterion()
    optimizer = model.optimizer()
    num_inputs = model.num_inputs

    model = model.to(device)

    for epoch in range(120):
        epoch_loss = 0.0
        for i, data in enumerate(train_loader, 0):
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

        epoch_loss /= len(train_loader) / 64

        epoch_plus_one = epoch + 1
        print(f'[Epoch: {epoch_plus_one}] loss: {epoch_loss:.5f}')

        if epoch_plus_one > 4:
            # Save epochs for each iteration after the first couple of epochs have passed
            save_model(model, path_prefix + f"{epoch_plus_one:03}" + ".onnx", num_inputs)


def main():
    # Make the training process deterministic
    torch.manual_seed(0)

    path = "./training-data/"
    Path(path).mkdir(exist_ok=True)

    mode = "contact"
    match mode:
        case "contact":
            num_inputs = 202
            model = Model(num_inputs)
            train(model, path + "contact-inputs.csv", path + mode)
        case "race":
            # `Race` has fewer inputs than `Contact`
            num_inputs = 186
            model = Model(num_inputs)
            train(model, path + "race-inputs.csv", path + mode)
        case "tiny_race":
            # This is used to be committed to the repository and not taking up much space.
            num_inputs = 186
            model = TinyModel(num_inputs)
            train(model, path + "race-inputs.csv", path + mode)
        case _:
            print("Invalid mode")

    print('Finished Training')


if __name__ == "__main__":
    main()
