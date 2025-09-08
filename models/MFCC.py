import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from assets import RESULTS_DIR

class MFCCDataset(Dataset):
    def __init__(self, split="train", label2idx=None):
        self.split = split
        self.data = []
        self.labels = []
        self.label2idx = label2idx or {}
        self.idx2label = {}

        split_dir = os.path.join(RESULTS_DIR, "features", split)
        classes = sorted(os.listdir(split_dir))

        if not self.label2idx and split == "train":
            for idx, label in enumerate(classes):
                self.label2idx[label] = idx

        for label in classes:
            class_dir = os.path.join(split_dir, label)
            for file in os.listdir(class_dir):
                if not file.endswith(".npy"):
                    continue
                mfcc = np.load(os.path.join(class_dir, file))

                if label in self.label2idx:
                    idx = self.label2idx[label]
                else:
                    idx = -1  # mark as OOSR

                self.data.append(mfcc.T)
                self.labels.append(idx)

        self.idx2label = {v: k for k, v in self.label2idx.items()}


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def collate_fn(batch):
    xs, ys = zip(*batch)
    xs_padded = pad_sequence(xs, batch_first=True) 
    ys = torch.tensor(ys, dtype=torch.long)
    return xs_padded, ys

def get_dataloaders(batch_size=4):
    train_set = MFCCDataset("train")

    val_set_full = MFCCDataset("val", label2idx=train_set.label2idx)
    val_indices = [i for i, y in enumerate(val_set_full.labels) if y != -1]

    val_set_full.data = [val_set_full.data[i] for i in val_indices]
    val_set_full.labels = [val_set_full.labels[i] for i in val_indices]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set_full, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, train_set.label2idx, train_set.idx2label
