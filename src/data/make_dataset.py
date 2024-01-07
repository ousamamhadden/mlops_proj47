import glob
import os
import os.path as path

import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import Normalize

dts = ["train", "test"]
for dt in dts:
    # create empty lists to store the train/test images and targets
    images = []
    targets = []
    # collect all files for training/testing
    images_file_paths = sorted(glob.glob(path.join(os.getcwd(), f"data/raw/{dt}_images*.pt")))
    targets_file_paths = sorted(glob.glob(path.join(os.getcwd(), f"data/raw/{dt}_target*.pt")))
    # loop through the train/test files and load them into the lists
    for i in range(len(images_file_paths)):
        images.append(torch.load(images_file_paths[i]))
        targets.append(torch.load(targets_file_paths[i]))
    # concatenate the lists into tensors
    images = torch.cat(images)
    targets = torch.cat(targets)
    # create a Normalize transform object
    normalize = Normalize(torch.mean(images), torch.std(images))
    # apply the transform to the tensor
    normalized_images = normalize(images)
    # create a TensorDataset from the normalised train/test images and targets
    train_set = TensorDataset(normalized_images.unsqueeze(1), targets)
    # save it
    torch.save(train_set, path.join(os.getcwd(), f"data/processed/{dt}_set.pt"))
