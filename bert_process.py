import pandas as pd
import re
import os
import torch
import torch.nn as nn
import numpy as np
import shutil
import sys
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from torch.utils.data import DataLoader
from sklearn import metrics
from transformers import BertModel
from transformers import BertTokenizer
from torch.utils.data import Dataset
from langdetect import detect
import preprocess

MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 32
BATCH_SIZE = 30
EPOCHS = 8
LEARNING_RATE = 0.0001
FILE_PATH = '/Users/nelesassor/Desktop/TwitterAnalyse/src/Model/model.pth'
TRAIN_SIZE = 0.8

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

ohe_df = preprocess.df1


class TweetDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe['Clean_Content']
        self.max_len = max_len
        # check if dataframe has 'Label' column
        if 'Label' in dataframe.columns:
            self.labels = dataframe['Label']
        else:
            self.labels = None

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        result = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
        # add labels to result if they exist
        if self.labels is not None:
            result['targets'] = torch.tensor(self.labels[index], dtype=torch.float)

        return result


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.l1 = BertModel.from_pretrained(MODEL_NAME, return_dict=False)
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def train_model(model, training_loader, device):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        for batch_idx, data in enumerate(training_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')


def validate_model(model, validation_loader, device):
    model.eval()
    val_targets = []
    val_outputs = []
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    val_preds = (np.array(val_outputs) > 0.5).astype(int)
    accuracy = metrics.accuracy_score(val_targets, val_preds)
    f1 = metrics.f1_score(val_targets, val_preds, average='micro')

    print("Accuracy: " + str(accuracy))
    print("F1 Score: " + str(f1))


def predict(model, data_loader):
    model.eval()
    pred_outputs = []
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            outputs = model(ids, mask, token_type_ids)
            pred_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    return pred_outputs


# Split the data
train_dataset = ohe_df.sample(frac=TRAIN_SIZE, random_state=200)
valid_dataset = ohe_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

# Create datasets and data loaders
training_set = TweetDataset(train_dataset, tokenizer, MAX_LEN)
validation_set = TweetDataset(valid_dataset, tokenizer, MAX_LEN)
training_loader = DataLoader(training_set, batch_size=BATCH_SIZE)
validation_loader = DataLoader(validation_set, batch_size=BATCH_SIZE)

# Initialize the model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = torch.nn.BCEWithLogitsLoss()
model = BERT().to(device)

# Train and validate the model
train_model(model, training_loader, device)
validate_model(model, validation_loader, device)

# Save the model
torch.save(model.state_dict(), FILE_PATH)

# Predict with test data
testing_set = TweetDataset(valid_dataset, tokenizer, MAX_LEN)
testing_loader = DataLoader(testing_set, batch_size=BATCH_SIZE)
predictions = predict(model, testing_loader)
