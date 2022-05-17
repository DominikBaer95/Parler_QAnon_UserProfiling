import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from joblib import Parallel, delayed
import multiprocessing
import nums_from_string
from itertools import chain

model = SentenceTransformer('all-MiniLM-L6-v2')

# Define path
# Local
path_input = "..\\..\\..\\Data\\input\\parleys_user\\clean_text\\"
path_output = "..\\..\\..\\Data\\output\\embeddings_raw\\"

# List of files with content by users
file_list_content = os.listdir(path_input)

# Processed files
file_list_processed = os.listdir(path_output)

# Unprocessed files
all_files = set(chain.from_iterable([nums_from_string.get_nums(s) for s in file_list_content]))
processed = set(chain.from_iterable([nums_from_string.get_nums(s) for s in file_list_processed]))
file_list_unprocessed = ["user_content_" + str(i) + "_clean_embeddings.csv" for i in all_files - processed]


def compute_embeddings(filename):
    # Number of file
    file_number = nums_from_string.get_nums(filename)[0]
    # Path
    file_path = path_input + filename
    # Load file
    dta = pd.read_csv(file_path)
    # Filter entries which are no strings (i.e., nan)
    strings = dta["body"].apply(lambda x: type(x) == str)
    dta = dta[strings]
    # Create list of sentences
    sentences = list(dta.body)
    # Compute embeddings
    embeddings = model.encode(sentences)
    # Convert np array to pd data frame
    embeddings_df = pd.DataFrame(embeddings)
    # Merge embeddings with creator
    embeddings_df['creator'] = dta.creator
    # Save embeddings as csv
    embeddings_df.to_csv(path_output + "embeddings_" + str(file_number) + ".csv", encoding="utf-8")

    print("Processed file " + str(file_number) + " of " + str(len(file_list_content)))


print(multiprocessing.cpu_count())
Parallel(n_jobs=multiprocessing.cpu_count() - 1, backend="loky", verbose=10)(
    delayed(compute_embeddings)(file) for file in file_list_unprocessed)
