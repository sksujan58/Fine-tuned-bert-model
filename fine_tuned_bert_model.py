import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model_name = "bert-base-uncased"
# max sequence length for each document/sentence sample
max_length = 512
# load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

dataset = fetch_20newsgroups(subset="all", shuffle=True, remove=("headers", "footers", "quotes"))

documents = dataset.data

labels = dataset.target

import pandas as pd

df = pd.DataFrame({"documents": documents, "labels": labels})

import re, string


def text_cleaner(text):
    '''some text cleaning method'''

    text = text.lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text


newdf = df.copy()

newdf["documents"] = newdf["documents"].apply(text_cleaner)


def split(dataset, test_size=0.2):
    documents = dataset.documents
    labels = dataset.labels
    # split into training & testing a return data as well as label names
    return train_test_split(documents, labels, test_size=test_size)


(train_texts, valid_texts, train_labels, valid_labels) = split(newdf)

# using only 1000 dataset for training
# and 100 dataset for testing

(train_texts, valid_texts, train_labels, valid_labels) = (
list(train_texts[:1000]), list(valid_texts[:100]), train_labels[:1000].values, valid_labels[:100].values)

train_labels = train_labels.astype(torch.LongTensor)
valid_labels = valid_labels.astype(torch.LongTensor)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)


class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=20).to(device)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }


training_args = TrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,
    # batch size for evaluation               # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay            # directory for storing logs
    load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
    # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
    evaluation_strategy="epoch",
    save_strategy="epoch"  # evaluate each `logging_steps`
)

trainer = Trainer(
    model=model,  # the instantiated Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=valid_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,  # the callback that computes metrics of interest
)

trainer.train()
