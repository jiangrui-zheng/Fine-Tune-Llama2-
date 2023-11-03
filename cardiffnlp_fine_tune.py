import sys
import os
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import argparse


def train_hate_model(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    conan = pd.read_csv("/data/shared/hate_speech_dataset/CONAN.csv")
    hate_df = conan[['hateSpeech']].drop_duplicates()
    hate_df = hate_df.rename(columns={'hateSpeech': 'sentence'})
    hate_df = hate_df.reset_index(drop=True)
    hate_df['labels'] = 1

    nonhate_df = conan[['counterSpeech']].drop_duplicates()
    nonhate_df = nonhate_df.rename(columns={'counterSpeech': 'sentence'})
    nonhate_df = nonhate_df.reset_index(drop=True)
    nonhate_df['labels'] = 0
    train_df = pd.concat([hate_df, nonhate_df], ignore_index=True)

    compare_datasets = "/data/jzheng36/hatemoderate/hatemoderate/fine_tune/cardiffnlp.pkl"

    model_args = ClassificationArgs()
    model_args.learning_rate = args.learning_rate
    model_args.num_train_epochs = args.n_epoch
    model_args.train_batch_size = 32
    model_args.eval_batch_size = 32
    model_args.n_gpu = 4
    model_args.output_dir = "{}_lr={}_epoch={}_hatemoderate".format(args.model_name.replace("/", "-"),
                                                                    args.learning_rate, args.n_epoch)
    model_args.overwrite_output_dir = True
    model_args.save_best_model = True
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.evaluate_during_training = False

    model = ClassificationModel(args.model_type, args.model_name, num_labels=2, args=model_args)

    cardiffnlp_datasets = pd.read_pickle(compare_datasets)
    cardiffnlp_datasets = cardiffnlp_datasets.rename(columns={"label": "labels"})
    cardiffnlp_datasets = cardiffnlp_datasets[cardiffnlp_datasets['split'] != 'test']

    columns = ["text", "labels"]

    train_df = train_df.rename(columns={"sentence": "text"}).sample(frac=1)
    if args.include == True:
        train_df = pd.concat([train_df[columns], cardiffnlp_datasets[columns]])
    else:
        train_df = pd.concat([train_df[columns]])

    model.train_model(train_df=train_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a hate speech classification model.")
    parser.add_argument("--model_name", type=str, help="The name or path of the pre-trained model.")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="The learning rate to use for training. Default: 5e-6.")
    parser.add_argument("--n_epoch", type=int, default=3, help="The number of epochs to train for. Default: 3.")
    parser.add_argument("--model_type", type=str, default="roberta",
                        help="The type of the model (e.g., 'roberta', 'bert'). Default: 'roberta'.")
    parser.add_argument("--include", action="store_true", default=True,
                        help="Whether to include the compared dataset in training. Default: True.")
    parser.add_argument("--no-include", action="store_false", dest="include",
                        help="Do not include the compared dataset in training.")
    args = parser.parse_args()
    train_hate_model(args)
