import os
import hydra
from happytransformer import HappyTextToText, TTTrainArgs,TTSettings
import wandb

os.environ["WANDB_PROJECT"] = "mlops-proj47"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "ValidationLoss"},
    "parameters": {
        "batch_size": {"max": 16, "min": 1},
        "learning_rate": {"max": 0.01, "min": 0.00000001},
        "num_train_epochs": {"max": 3, "min": 1},
        "eval_ratio": {"max": 0.3, "min": 0.01}
    },
}
#@hydra.main(config_path='models/config', config_name="config.yaml")
def train(config):
    model = HappyTextToText('t5-small')
    args = TTTrainArgs(batch_size=config.batch_size, report_to="wandb",project_name="mlops-47",run_name="test",learning_rate=config.learning_rate,num_train_epochs=config.num_train_epochs,)
    beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=100)
    input_text_1 = "grammar: I I wants to codes."
    output_text_1 = model.generate_text(input_text_1, args=beam_settings)
    print(output_text_1.text)
    model.train('data/processed/train.csv', args=args)
    output_text_1 = model.generate_text(input_text_1, args=beam_settings)
    print(output_text_1.text)
    model.save("model/")
    res = model.eval('data/processed/train.csv')
    wandb.log({"ValidationLoss": res.loss})
    


def main():
    train(wandb.config)


if __name__=='__main__':
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="mlops-47")
    wandb.agent(sweep_id, function=main, count=10)
