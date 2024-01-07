import os
import os.path as path

import click
import matplotlib.pyplot as plt
import torch

import wandb
from models.model import MyAwesomeModel


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    data = torch.cat([batch[0] for batch in dataloader], 0)
    labels = torch.cat([batch[1] for batch in dataloader], 0)
    return model(data), data, labels


@click.command()
@click.argument("trained_model", type=click.Path(exists=True))
@click.argument("data", type=click.Path(exists=True))
def main(trained_model, data):
    wandb.init(project="project_1")
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(path.join(os.getcwd(), trained_model)))
    test_set = torch.load(path.join(os.getcwd(), data))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    model.eval()

    with torch.no_grad():
        logits, images, labels = predict(model, test_loader)
        ps = torch.exp(logits)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))
    print(f"Total Accuracy: {accuracy.item()*100}%")

    indices = torch.randperm(labels.shape[0])[:10]
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    images_t = images[indices]
    predictions_t = labels[indices]
    # wandb.log({"examples": [wandb.Image(im) for im in images_t]})

    # my_table = wandb.Table()
    # my_table.add_column("image", images_t.numpy())
    # my_table.add_column("class_prediction", predictions_t.numpy())

    table = wandb.Table(columns=["Image", "Prediction"])

    for img, label in zip(images_t, predictions_t):
        table.add_data(wandb.Image(img), label)
    wandb.log({"Table": table})

    # # Log your Table to W&B
    # wandb.log({"mnist_predictions": my_table})

    for ax, idx in zip(axes.flatten(), indices):
        image = images[idx]
        label = labels[idx]
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(label.item())
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(path.join(os.getcwd(), "reports/figures/sample_predictions.png"), dpi=200)

    wandb.finish()


if __name__ == "__main__":
    main()
