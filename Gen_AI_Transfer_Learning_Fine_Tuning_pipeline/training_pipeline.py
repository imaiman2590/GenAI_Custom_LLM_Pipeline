from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq, GPT2LMHeadModel, EarlyStoppingCallback
)
from datasets import Dataset
import pandas as pd

TRAINING_PARAMS = {
    "output_dir": "./results",
    "learning_rate": 3e-5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "num_train_epochs": 5,
    "weight_decay": 0.01,
    "save_total_limit": 1,
    "fp16": True,
    "gradient_accumulation_steps": 4,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "logging_steps": 50,
    "save_steps": 100,
    "load_best_model_at_end": True,
    "metric_for_best_model": "loss",
    "save_strategy": "steps",
    "report_to": "none",
    "dataloader_num_workers": 4
}

def fine_tune_text_model(data: pd.DataFrame, model_name: str, output_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "gpt2" in model_name.lower():
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def tokenize(examples):
        inputs = tokenizer(examples["source"], truncation=True, padding="max_length", max_length=128)
        targets = tokenizer(examples["target"], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = targets["input_ids"]
        return inputs

    dataset = Dataset.from_pandas(data)
    tokenized_dataset = dataset.map(tokenize, batched=True)

    # Split into train and eval sets (e.g., 90% train / 10% eval)
    train_test = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        **TRAINING_PARAMS
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
