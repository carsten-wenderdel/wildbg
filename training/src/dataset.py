from torch.utils.data import Dataset
import torch

class WildBgDataSet(Dataset):
    def __init__(self, csv_files: list | str):
        # If you want to combine several CSV files from another folder, use the following: 
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        labels = []
        inputs = []
        for path in csv_files:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:
                    line = line.strip().split(';')
                    line = list(map(float, line))
                    labels.append(line[:6])
                    inputs.append(line[6:])
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # First 6 columns are outputs, last 202 columns are inputs
        return self.inputs[idx], self.labels[idx]
