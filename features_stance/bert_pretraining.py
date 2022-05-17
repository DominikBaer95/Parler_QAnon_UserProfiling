"""
    Pre-training of BERT (i.e. domain adaptation for Parler)
"""

from datasets import load_dataset
import random
from typing import Dict

from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling

import torch
torch.cuda.device_count()

# Language model

card = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(card, use_fast=True)
model = AutoModelForMaskedLM.from_pretrained(card)

# Load datasets
datasets = load_dataset("text", data_files={"train": "train_text.txt",
                                            "validation": "test_text.txt"})


# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_datasets = datasets.map(tokenize_function)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="model/ParlerBERT",
    overwrite_output_dir=True,
    #max_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=24,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    #save_steps=50,
    #save_total_limit=2,
    #logging_steps=50,
    seed=42,
    fp16=True,
    dataloader_num_workers=8,
    load_best_model_at_end=True,
    no_cuda=False,
    gradient_accumulation_steps=42,
    num_train_epochs=2,
    learning_rate=0.0005,
    weight_decay=0.01,
    logging_dir='./',
    logging_steps=42
)


class LoggerTrainer(Trainer):
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log :obj:`logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

        with open('lossLog.txt', 'a') as file:
            file.write(str(output) + '\n')


trainer = LoggerTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model("model/bert_parler")


