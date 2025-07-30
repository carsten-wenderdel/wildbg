from abc import abstractmethod, ABC
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch


class WildBgDataSet(Dataset):
    def __init__(self, csv_files: list | str):
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        labels = []
        inputs = []
        for path in csv_files:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip().split(',')
                    line = list(map(float, line))
                    labels.append(line[:6])
                    inputs.append(line[6:])
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class Model(torch.nn.Module, ABC):
    # Override that in implementing classes
    num_inputs = 0

    @abstractmethod
    def optimizer(self):
        pass


# Wrap a model with logits as output and add softmax so that all outputs add up to 1.
class Wrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.base_model(x)
        return self.softmax(logits)


def save_model(model: nn.Module, path: str, num_inputs: int) -> None:
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
    criterion = nn.CrossEntropyLoss()
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
