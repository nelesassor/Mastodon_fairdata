import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re


files = ['../data/mastodon/python_tooswales_toots_fairdata_228.csv',
         '../data/mastodon/python_tooswales_toots_fairdata_228.csv',
         '../data/mastodon/python_science_Mastodon_toots_fairdata_205.csv',
        '../data/mastodon/python_nfdi_toots_fairdata_26.csv',
        '../data/mastodon/python_openbiblio_toots_fairdata_213.csv',
        '../data/mastodon/python_qotoorg_toots_fairdata_342.csv',
         '../data/mastodon/python_Scicommxyz_toots_fairdata_435.csv',
         '../data/mastodon/python_mstdnscience_toots_fairdata_290.csv',
         '../data/mastodon/python_newsiesocial_toots_fairdata_41.csv',
         '../data/mastodon/python_mastodonscot_toots_fairdata_394.csv',
         '../data/mastodon/python_med_mastodon_toots_fairdata_240.csv',
         '../data/mastodon/python_mastodonScienceCom_toots_fairdata_16_06_23_205.csv',
         '../data/mastodon/python_mastodonnz_toots_fairdata_5.csv',
         '../data/mastodon/python_mastodonied_toots_fairdata_319.csv',
         '../data/mastodon/python_mastodongreen_toots_fairdata_438.csv',
         '../data/mastodon/python_mastodongreen_toots_fairdata_438.csv',
         '../data/mastodon/python_mastodonau_toots_fairdata_153.csv',
         '../data/mastodon/python_mastodon_toots_fair-data_160623_500.csv',
         '../data/mastodon/python_homesocial_toots_fairdata_275.csv',
         '../data/mastodon/python_genomic_toots_fairdata_369.csv',
         '../data/mastodon/python_fediscience_toots_fairdata_466.csv',
         '../data/mastodon/python_fairpointssocial_toots_fairdata_70.csv',
         '../data/mastodon/python_ecoevo_toots_fairdata_417.csv',
         '../data/mastodon/python_chaossocial_toots_fairdata_443.csv',
         '../data/mastodon/python_aussocial_toots_fairdata_314.csv',
         '../data/mastodon/python_astrodon_toots_fairdata_137.csv',
         '../data/mastodon/python_appuk_toots_fairdata_392.csv'
        ]

# Empty list to store DataFrames
dataframes = []

for file in files:
    # Read each CSV file into a DataFrame and append it to the list
    dataframes.append(pd.read_csv(file))

# Concatenate all the dataframes in the list into a single DataFrame
df = pd.concat(dataframes, keys=files)


def preprocess_text(text):
    # HTML-Tags entfernen
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # URLs entfernen
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Nicht-alphabetische Zeichen entfernen und in Kleinbuchstaben umwandeln
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()

    # Mehrfache Leerzeichen entfernen
    text = re.sub(r'\s+', ' ', text).strip()

    return text