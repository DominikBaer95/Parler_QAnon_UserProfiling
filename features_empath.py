"""
    File to create Empath features (Fast, Chen, and Bernstein 2016)  for Parler content
"""

import os
import nums_from_string
import string
from itertools import chain
import pandas as pd
from empath import Empath
from joblib import Parallel, delayed
import multiprocessing

# Define path
path_input = "..\\..\\..\\Data\\input\\parleys_user\\"
path_output = "..\\..\\..\\Data\\output\\features_empath\\"


def create_empath_features(filename):
    # Load file
    dta = pd.read_csv(path_input + filename, usecols=["creator", "id", "body"])

    # File number
    file_number = nums_from_string.get_nums(filename)[0]

    # Convert text to lower case
    dta["body"] = dta["body"].str.lower()

    # Clean text
    dta["body"] = dta["body"].str.replace("@[a-z,A-Z,0-9]*", "", regex=True)  # Remove handles
    dta["body"] = dta["body"].str.replace("[^\x01-\x7F]", "", regex=True)  # Remove emojis
    dta["body"] = dta["body"].str.replace('[{}]'.format(string.punctuation), "", regex=True)  # Remove punctuation
    dta["body"] = dta["body"].str.replace("[^[:alnum:][:space:]]", "", regex=True)  # Remove all non-alpha-numeric characters and space
    dta["body"] = dta["body"].str.replace("\\s\\s+", "", regex=True)  # Remove whitespace

    # Analyze data with empath
    empath_features = dta["body"].apply(lambda x: pd.Series(lexicon.analyze(x, normalize=True), dtype=float))

    # Bind columns to original dataframe
    df = dta.join(empath_features, lsuffix="_left").drop("body_left", axis=1)

    # Save raw file
    df.to_csv(path_output + "raw\\" + "empath_features_" + str(file_number) + ".csv")

    # Aggregate by user and average
    df_agg = df.groupby("creator").mean()

    # Save features by user file
    df_agg.to_csv(path_output + "by_user\\" + "empath_features_" + str(file_number) + ".csv")

    print("Processed file " + str(file_number) + " of " + str(len(file_list_content)))


# Import empath lexicon
lexicon = Empath()

# List of files with content by users
file_list_content = os.listdir(path_input)

# Processed files
file_list_processed = os.listdir(path_output + "by_user\\")

# Unprocessed files
all_files = set(chain.from_iterable([nums_from_string.get_nums(s) for s in file_list_content]))
processed = set(chain.from_iterable([nums_from_string.get_nums(s) for s in file_list_processed]))
file_list_unprocessed = ["user_content_" + str(i) + ".csv" for i in all_files - processed]

print(multiprocessing.cpu_count())
Parallel(n_jobs=multiprocessing.cpu_count() - 1, backend="loky", verbose=10)(
    delayed(create_empath_features)(file) for file in file_list_unprocessed)

# Concatenate files and create final feature set
features_empath = pd.concat([pd.read_csv(path_output + "by_user\\" + file) for file in file_list_processed])
features_empath.to_csv("..\\..\\..\\Data\\output\\" + "features_empath.csv", index=False)

