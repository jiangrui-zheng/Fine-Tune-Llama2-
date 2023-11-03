import torch
from utils import process_dataset



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

datasets=("/data/shared/hate_speech_dataset/Hate_Check.csv"
          "/data/jzheng36/hatemoderate/hatemoderate/postprocess/test_hatE.csv"
          "/data/jzheng36/hatemoderate/hatemoderate/postprocess/test_htpo.csv"
          "/data/jzheng36/hatemoderate/hatemoderate/postprocess/test_hatex.csv")
models=("/data/jzheng36/model/card_only_roberta-base_lr=5e-06_epoch=2"
        "/data/jzheng36/model/roberta-base_lr=5e-06_epoch=2")

labels = "LABEL_1"


for dataset in datasets:
        for model in models:
                process_dataset(dataset, model, labels, device)
