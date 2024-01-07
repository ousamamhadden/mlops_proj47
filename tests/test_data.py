import os
import pytest
import torch


@pytest.mark.skipif(not (os.path.exists("data/processed/train_set.pt") and os.path.exists("data/processed/test_set.pt")), reason="Data files not found")
def test_data():
    train_set = torch.load("data/processed/train_set.pt")
    test_set = torch.load("data/processed/test_set.pt")
    assert len(train_set) == 30000 and len(test_set) == 5000, "Dataset did not have the correct number of samples"
    for img, target in train_set:
        assert img.shape == (1,28,28), "The shape does not equal to [1, 28, 28]"
    labels = torch.tensor([y for x, y in train_set])
    unique_labels = torch.unique(labels)
    unique_labels = unique_labels.tolist()
    expected_labels = [0,1,2,3,4,5,6,7,8,9]
    assert unique_labels == expected_labels, "Some of the lables are missing in the dataset"