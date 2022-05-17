"""
    Fine-tuning of BERT for stance detection
"""

import glob
import random
import re
import itertools

import numpy as np
import nums_from_string
import pandas as pd
import emoji

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import TextClassificationPipeline

from datasets import Dataset
from datasets import load_metric

from joblib import Parallel, delayed
import multiprocessing


random.seed(42)

# Define paths
path_odds_ratio = "..\\..\\..\\Data\\output\\odds_ratio\\"
path_content_qanon = "..\\..\\..\\Data\\input\\parleys_qanon\\"
path_features_stance = "..\\..\\..\\Data\\output\\features_stance\\"

"""
  Extend vocabulary as in Kawintiranon & Singh (2021) https://aclanthology.org/2021.naacl-main.376.pdf
"""

# Select stance tokens
# Load and concatenate txt files with top words from log_odd_ratio analysis
files_top_words = glob.glob(path_odds_ratio + "top_words_*")
top_words = list(itertools.chain.from_iterable([open(file, "r").readlines() for file in files_top_words]))

# Remove duplicates
top_words = list(set(top_words))

# Save stance tokens
with open(str(path_odds_ratio + "stance_tokens.txt"), "w", encoding="utf-8") as f:
    for word in top_words:
        f.write(word)

stance_tokens = [re.sub(r"\n", "", token) for token in top_words]

# Extend vocabulary
# Load pre-trained bert_parler model
card = "bert_parler"
tokenizer = AutoTokenizer.from_pretrained(card, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(card)

# Current vocab size
print("Current vocab size " + str(len(tokenizer)))

tokenizer.add_tokens(stance_tokens)
model.resize_token_embeddings(len(tokenizer))

# New vocab size
print("New vocab size " + str(len(tokenizer)))

tokenizer.save_pretrained("bert_parler_stance")
model.save_pretrained("bert_parler_stance")

"""
    Fine-tune bert_parler_stance for stance detection
"""

# Load labeled data
df = pd.read_csv("..\\..\\..\\Data\\output\\data_stance_finetuning.csv", usecols=["labels", "text"])

# Combine Oppose (2) and Neither (3) labels for classification
df.loc[df["labels"] == 2, ["labels"]] = 0
df.loc[df["labels"] == 3, ["labels"]] = 0
df["labels"] = df["labels"].astype("int64")

# Create train test split
dataset = Dataset.from_pandas(df).train_test_split(test_size=0.3)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Tokenize datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True).remove_columns("text")

# Load model for Classification
card = "bert_parler_stance"
model = AutoModelForSequenceClassification.from_pretrained(card, num_labels=2)

# Load evaluation metric ROC_AUC
metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Define training arguments
training_args = TrainingArguments(
    output_dir="stance_detection",
    #overwrite_output_dir=True,
    evaluation_strategy="epoch",
    #save_strategy="epoch",
    #dataloader_num_workers=7,
    #load_best_model_at_end=True,
    #no_cuda=True,
    #num_train_epochs=2,
    #learning_rate=0.0005,
    #weight_decay=0.01,
    #logging_dir='\\log',
    #logging_steps=42
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# Fine-tune model
trainer.train()

# Save fine-tuned model
trainer.save_model("bert_parler_stance_finetuned")

"""
    Predict stance of user data
"""


def create_stance_features(file):

    file_number = nums_from_string.get_nums(file)[0]

    #print("Processing file " + str(file_number))

    # Load and preprocess data
    df = pd.read_csv(file)
    df["body"] = df["body"].str.lower()
    df["body"] = df["body"].apply(lambda x: emoji.demojize(x)).str.replace(":[^: ]+:", " ", regex=True).str.replace(" +", " ", regex=True)

    # Predict stance on new samples
    pred = pipe(list(df["body"]))

    # Reformat predictions
    scores = pd.DataFrame([x[1] for x in pred]).join(df).drop(columns=["label", "body"]).rename(columns={"score": "stance_qanon"})

    # Save predicted stance
    scores.to_csv(path_features_stance + "raw\\features_stance_" + str(file_number) + ".csv", index=False)

    # Aggregate by user and average
    df_agg = scores.groupby("creator").mean()

    # Save features by user file
    df_agg.to_csv(path_features_stance + "\\by_user\\features_stance_" + str(file_number) + ".csv", index=True)

    print("Processed file " + str(file_number))


# File list of user data
file_list_user = glob.glob(path_content_qanon + "users_with_bios\\" + "*.csv")

# Processed files
file_list_processed = glob.glob(path_features_stance + "by_user\\" + "*.csv")

# Unprocessed files
all_files = set(itertools.chain.from_iterable([nums_from_string.get_nums(s) for s in file_list_user]))
processed = set(itertools.chain.from_iterable([nums_from_string.get_nums(s) for s in file_list_processed]))
file_list_unprocessed = [path_content_qanon + "users_with_bios\\user_content_qanon_" + str(i) + ".csv" for i in all_files - processed]

# Prepare prediction pipline
card = "bert_parler_stance_finetuned"
model = AutoModelForSequenceClassification.from_pretrained(card, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("bert_parler_stance")
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, framework="pt", return_all_scores=True)

# Compute predictions
print(multiprocessing.cpu_count())
Parallel(n_jobs=multiprocessing.cpu_count() - 1, backend="loky", verbose=10)(
    delayed(create_stance_features)(file) for file in file_list_unprocessed)

# Concatenate files and create final feature set
features_stance = pd.concat([pd.read_csv(file) for file in file_list_processed])
features_stance.to_csv("..\\..\\..\\Data\\output\\" + "features_stance.csv", index=False)
