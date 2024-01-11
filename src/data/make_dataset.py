import csv
from datasets import load_dataset

def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["input", "target"])
        for example in dataset:
            input_text = f"grammar: {example['input']}"
            target_text = example['output']
            writer.writerow([input_text, target_text])

def main(csv_path, number_of_examples):
    dataset_train = load_dataset('liweili/c4_200m', split='train', streaming=True)
    dataset_train = dataset_train.take(number_of_examples)
    generate_csv(csv_path, dataset_train)

if __name__=='__main__':
    PATH='data/processed/train.csv'
    main(PATH, 100)
