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


path='../data'


def pre_process(text):
#    text = BeautifulSoup(text).get_text()
    text = BeautifulSoup(text, features="html.parser").get_text()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    tokens = text.split()
    return " ".join(tokens)

def filter_non_english_tweets(dataframe, lang='en'):
    filtered_df = dataframe[dataframe['Clean_Content'].apply(lambda x: detect(x) == lang)]
    return filtered_df
def clean_category(category):
    category_dict = {'Veranstaltung, Publikation ': 'Veranstaltung',
                     'Meinung ': 'Meinung',
                     'Dataset': 'Divers',
                     'Job': 'Divers',
                     'Initiative': 'Divers',
                     'Veranstaltung ': 'Veranstaltung'}

    if category in category_dict:
        return category_dict[category]
    elif category == 'Divers':
        return None
    else:
        return category


def clean_dataframe(df):
    # Drop null values in 'Kategorisierung' column
    df = df[df['Kategorisierung'].notna()]

    # Clean 'Kategorisierung' column
    df.loc[:, 'Kategorisierung'] = df['Kategorisierung'].apply(clean_category)

    # Drop rows where 'Kategorisierung' is None
    df = df[df['Kategorisierung'].notna()]

    # Pre-process 'Content' column
    df['Clean_Content'] = df['Content'].apply(pre_process)

    # Filter non-English tweets
    df = filter_non_english_tweets(df)

    return df

df_manuell = pd.read_excel('../data/manuell_kategorisiert/pyhon_tweets_fairdata_Kategorisierung_ManuellExcel.xlsx')
df_manuell.columns = ['Data', 'ID', 'Content', 'Kategorisierung', 'Username', 'LikeCount', 'RetweetCount']

# Exclude first row
df1 = df_manuell.iloc[1:]

# Apply cleaning functions
df1 = clean_dataframe(df1)


def cleaning_text(text):
    text = text.lower()
    text = re.sub('r<.*?>', ' ', text)
    text = re.sub(r'#\w+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'http\S+', " ", text)
    text = text.split()
    stop_words = stopwords.words("english")
    text = " ".join([word for word in text if not word in stop_words])
    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")
    return text

def process_dataframe(df):
    df['Clean_Content'] = df['Clean_Content'].apply(pre_process)
    df = filter_non_english_tweets(df)
    df = df.loc[:, ['Clean_Content','Kategorisierung']]
    test_dataset = df[170:]
    df = df[:170]
    one_hot_encoded = pd.get_dummies(df['Kategorisierung'])
    df = pd.concat([df, one_hot_encoded], axis=1)
    df = df.drop('Kategorisierung', axis=1)
    df['Label'] = df[['Meinung', 'Publikation', 'Veranstaltung']].values.tolist()
    df = df.drop(columns = ['Meinung', 'Publikation', 'Veranstaltung'])
    df['Clean_Content'] = df['Clean_Content'].apply(cleaning_text)
    return df, test_dataset


df1.columns = ['Data', 'ID', 'Content', 'Kategorisierung', 'Username', 'LikeCount', 'RetweetCount', 'Clean_Content']
df1 = process_dataframe(df1)

