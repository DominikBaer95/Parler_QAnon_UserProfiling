# Using this file we compute text embeddings based on SBERT (Reimers & Gurevych 2019)
# Iteratively, the cleaned files containing all posts and comments for a subsample of users are loaded and text embeddings are computed

import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed
import multiprocessing
import nums_from_string
from itertools import chain
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Define path
# Local
path_input = "..\\data\\input\\parleys_user\\clean_text\\"
path_output = "..\\data\\output\\embeddings_raw\\"

# List all files containing cleaned parler posts and comments
file_list_content = os.listdir(path_input)
print(file_list_content)

files = list(chain.from_iterable([nums_from_string.get_nums(s) for s in file_list_content]))
files_str = [str(i) for i in files]
file_names = ["user_content_" + s + "_clean_embeddings.csv" for s in files_str]
print(file_names)

def compute_embeddings(filename):
    # Number of file
    file_number = [int(i) for i in filename.split("_") if i.isdigit()][0]
    # Path
    file_path = path_input + filename
    # Load file
    dta = pd.read_csv(file_path)
    sentences = list(dta.body)
    # Compute embeddings
    embeddings = model.encode(sentences)
    # Convert np array to pd data frame
    embeddings_df = pd.DataFrame(embeddings)
    # Merge embeddings with creator
    embeddings_df['creator'] = dta.creator
    # Save embeddings as csv
    embeddings_df.to_csv(path_output + "embeddings_" + str(file_number) + ".csv", encoding="utf-8")


print(multiprocessing.cpu_count())
Parallel(n_jobs=multiprocessing.cpu_count() - 1, backend="loky", verbose=10)(delayed(compute_embeddings)(file) for file in file_names)