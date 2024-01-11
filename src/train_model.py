from datasets import load_from_disk
from happytransformer import HappyTextToText, TTTrainArgs

def train(train_data_path):
    model = HappyTextToText('t5-small')
    args = TTTrainArgs(batch_size=8)

    model.train(train_data_path, args=args)

if __name__=='__main__':
    PATH='data/processed/train.csv'
    train(PATH)
