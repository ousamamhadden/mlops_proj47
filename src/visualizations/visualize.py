import os
import os.path as path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from ..models.model import MyAwesomeModel


def hook_fn(module, input, output):
    """A hook function that is used to extract features from a convolutional network in PyTorch.

    This function is registered to a layer of the network using the register_forward_hook method, and is called after the layer's forward pass. It appends the output tensor of the layer to a global list called features.

    Parameters
    ----------
    module : torch.nn.Module
      The module (layer) that this hook is attached to.
    input : tuple of torch.Tensor
      The input tensors to the module.
    output : torch.Tensor
      The output tensor of the module.

    Returns
    -------
    None
    """
    global features
    features.append(output)


@click.command()
@click.argument("trained_model", type=click.Path(exists=True))
def main(trained_model):
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(path.join(os.getcwd(), trained_model)))
    model.conv3.register_forward_hook(hook_fn)
    test_set = torch.load(path.join(os.getcwd(), "data/processed/train_set.pt"))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    model.eval()

    with torch.no_grad():
        _ = torch.cat([model(images) for images, _ in test_loader], 0)
        labels = torch.cat([labels for _, labels in test_loader], 0)

    features_ = np.concatenate(features, axis=0)

    features_ = features_.reshape(features_.shape[0], -1)

    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_)

    plt.figure(figsize=(10, 10))
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="rainbow")
    plt.colorbar()
    plt.title("t-SNE visualization of CNN features")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(path.join(os.getcwd(), "reports/figures/t_sne_reduced.png"), dpi=200)


if __name__ == "__main__":
    features = []
    main()
