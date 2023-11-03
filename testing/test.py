import torch
from utils import process_dataset



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# datasets=("hatemoderate.csv",
#           '/data/jzheng36/HateModerate/fine_tune/datasets/testing/Hate_Check.csv')
dataset="hatemoderate.csv"
#models=("/data/jzheng36/Fine-Tune-Llama2-/cardiffnlp-twitter-roberta-base-hate-latest_lr=5e-06_epoch=3_hatemoderate",
        #"/data/jzheng36/Fine-Tune-Llama2-/llama2_ft_7B-lr=2e-4")

model=("/data/jzheng36/Fine-Tune-Llama2-/models_7b")

labels = "LABEL_1"


# for dataset in datasets:
#         for model in models:
#                 process_dataset(dataset, model, labels, device)

#for model in models:
        #process_dataset(dataset, model, labels, device)


process_dataset(dataset, model, labels, device)