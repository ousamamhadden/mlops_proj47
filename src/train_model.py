import os
import os.path as path

import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer

from models.model import MyAwesomeModel


def main():
    print("Training day and night")

    model = MyAwesomeModel()

    epochs = 20
    trainer = Trainer(max_epochs=epochs)
    trainer.fit(model)

    plt.plot(list(range(1, epochs + 1)), model.train_losses)
    plt.title("Training results")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(path.join(os.getcwd(), "reports/figures/training_results.png"), dpi=200)

    trainer.test(model)

    torch.save(model.state_dict(), path.join(os.getcwd(), "models/trained_model.pt"))


if __name__ == "__main__":
    main()
