from transformers import LlamaForCausalLM, LlamaTokenizer
import transformers
import re
import pandas as pd
import torch
from tqdm import tqdm


with open('twitter_policies.txt', 'r') as file:
    content = file.read()

model_path = "/data/shared/llama2/llama/7B-Chat/"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)
pipeline = transformers.pipeline("text-generation", model=model, tokenizer = tokenizer)


def extract_category(text):
    patterns = [
        r'Category: \[(.*?)\]',
        r"\\n \\n Category:(.*?) \\n \\n"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    return text


def classify_by_twitter(sentence):
    res = pipeline(content + " \\n \\n Sentence:[" + sentence + "]", max_length=800)

    text = res[0]['generated_text']
    category = extract_category(text)
    return category


df = pd.read_csv("../postprocess/all_examples_0601_hate.csv", sep='\t')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

labels = []
for sentence in tqdm(df["sentence"]):
    label = classify_by_twitter(sentence)
    print(label)
    labels.append(label)

df["twitter"] = labels
df.to_csv("updated_all_examples_0601_hate.csv", sep='\t', index=False)