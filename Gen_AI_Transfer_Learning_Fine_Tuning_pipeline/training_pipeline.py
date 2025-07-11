from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq, GPT2LMHeadModel, EarlyStoppingCallback
)
from datasets import Dataset
import pandas as pd
import torch 

device = "GPU" if torch.cuda.is_available() else "CPU"
print(f"[INFO] Training will run on: {device}")

TRAINING_PARAMS = {
    "output_dir": "./results",
    "learning_rate": 3e-5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 4,
    "num_train_epochs": 5,
    "weight_decay": 0.01,
    "save_total_limit": 1,
    "fp16": torch.cuda.is_available(), 
    "gradient_accumulation_steps": 4,
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "logging_steps": 50,
    "logging_first_step": True,
    "save_steps": 100,
    "load_best_model_at_end": True,
    "metric_for_best_model": "loss",
    "save_strategy": "steps",
    "report_to": "none",
    "dataloader_num_workers": 4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "predict_with_generate": True,        
    "group_by_length": True,              
    "remove_unused_columns": True,
    "label_smoothing_factor": 0.1,
    "overwrite_output_dir": True,
    "disable_tqdm": False                 
}

def fine_tune_text_model(data: pd.DataFrame, model_name: str, output_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Handle GPT2-specific tokenizer padding issue
    if "gpt2" in model_name.lower():
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Tokenization function
    def tokenize(examples):
        inputs = tokenizer(examples["source"], truncation=True, padding="max_length", max_length=128)
        targets = tokenizer(examples["target"], truncation=True, padding="max_length", max_length=128)
        inputs["labels"] = targets["input_ids"]
        return inputs

    # Dataset preparation
    dataset = Dataset.from_pandas(data)
    tokenized_dataset = dataset.map(tokenize, batched=True)

    # Train-test split (90% train / 10% eval)
    train_test = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    # TrainingArguments setup
    training_args = TrainingArguments(
        output_dir=output_dir,
        **TRAINING_PARAMS
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Training
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
