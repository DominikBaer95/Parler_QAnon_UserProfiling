"""
    File to preprocess parleys for pre-training of BERT
    We follow https://aclanthology.org/2021.naacl-main.376.pdf and sample 5 Mio parleys for pre-training
    The preproccessing steps are adapted form https://aclanthology.org/2020.emnlp-demos.2/
"""

import glob
import random
import emoji
import numpy as np
import pandas as pd
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from nltk.tokenize import TweetTokenizer
import nums_from_string
from sklearn.model_selection import train_test_split

random.seed(42)


def get_lang_detector(nlp, name):
    return LanguageDetector()


def get_english_text(parley):
    doc = nlp(parley)
    lan = doc._.language.values()
    if "en" in lan:
        return parley
    else:
        return np.nan


def filter_tokens(parley):
    tt = TweetTokenizer()
    tokens = tt.tokenize(parley)
    if 10 < len(tokens) < 64:
        return parley
    else:
        return np.nan


# Path to unprocessed content data
path_content = "..\\..\\..\\Data\\input\\parler_data\\"
# Path to pre-training data
path_pretraining = "..\\..\\..\\Data\\input\\parleys_bert_pretraining\\"

# List raw files
file_list_content = glob.glob(path_content + "*.csv")

# Initialize language detection
nlp = spacy.load('en_core_web_sm')
Language.factory("language_detector", func=get_lang_detector)
nlp.add_pipe("language_detector", last=True)

# Clean parleys to collect a dataset of 5 Mio parleys for pre-training
# Initialize counter
i = 0

for k, file in enumerate(file_list_content):
    # Create dataframe of texts
    text = pd.read_csv(file, usecols=["body"])

    # File number
    file_number = nums_from_string.get_nums(file)[0]

    # Preprocessing
    # Filter empty Parleys
    text = text.dropna()

    # Drop duplicates (i.e. reposts)
    text = text.drop_duplicates()

    # Filter non-english text
    text["body"] = text["body"].apply(lambda x: get_english_text(x))
    text = text.dropna()

    # Filter parleys with less than 10 and more than 64 tokens
    text["body"] = text["body"].apply(lambda x: filter_tokens(x))
    text = text.dropna()

    # Translate emoji icons into text strings
    text["body"] = text["body"].apply(lambda x: emoji.demojize(x))

    # Substitute mentions with special token [ParlerUser]
    text["body"] = text["body"].str.replace(pat="@[a-z,A-Z,0-9]*", repl="[ParlerUser]", regex=True)

    text.to_csv(
        path_pretraining + "parleys_pretraining_" + str(file_number) + ".csv")

    print("Processed file " + str(file_number))
    i += len(text.index)

    print("Total number of parleys = " + str(i))

    # Break if 5000000 parleys are processed
    if i > 5000000:
        break

# List of input files
files_pretraining = glob.glob(path_pretraining + "*.csv")

# Combine files
parleys_pretraining = pd.concat([pd.read_csv(file) for file in files_pretraining]).drop("Unnamed: 0", axis=1)

# Drop duplicates (i.e. reposts)
parleys_pretraining = parleys_pretraining.drop_duplicates()

# Remove linebreaks
parleys_pretraining["body"] = parleys_pretraining["body"].str.replace("\n", " ")
# To lower
parleys_pretraining["body"] = parleys_pretraining["body"].str.lower()

# Create train and test split
train_text, test_text = train_test_split(parleys_pretraining["body"].tolist(), test_size=0.2, random_state=42)

with open(str(path_pretraining + "final\\train_text.txt"), 'w', encoding="utf-8") as file:
    for paragraph in train_text:
        file.write(paragraph + "\n")

with open(str(path_pretraining + "final\\test_text.txt"), 'w', encoding="utf-8") as file:
    for paragraph in test_text:
        file.write(paragraph + "\n")
