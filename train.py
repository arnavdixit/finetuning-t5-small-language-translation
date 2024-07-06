from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

import torch
from torch.utils.data import random_split

from config import get_config

def get_ds(config):
    raw_ds = load_dataset(config['datasource'], split = "train")
    return raw_ds

def run_training(config):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    ds = get_ds(config)
    def tokenize_function(examples):
        examples = examples['translation']
        inputs = [ex[config['src_lang']] for ex in examples]
        targets = [ex[config['tgt_lang']] for ex in examples]
        
        model_inputs = tokenizer(inputs, padding = "max_length", truncation = True)
        labels = tokenizer(text_target = targets, padding = "max_length", truncation = True)
            
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    ds = ds.map(tokenize_function, batched = True)
    train_ds_size = int(len(ds) * 0.9)
    valid_ds_size = len(ds) - train_ds_size
    
    train_ds, valid_ds = random_split(ds, [train_ds_size, valid_ds_size])
    print("Data Ready for training.")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir = config['output_dir'],
        eval_strategy = "epoch",
        learning_rate = config['lr'],
        per_device_train_batch_size = config['batch_size'],
        per_device_eval_batch_size = config['batch_size'],
        num_train_epochs = config['epochs'],
        weight_decay = 0.01,
        save_total_limit = 3,
        predict_with_generate = True,
    )
    
    trainer = Seq2SeqTrainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = valid_ds,
        tokenizer = tokenizer
    )
    print("Training Starting")
    trainer.train()
    results = trainer.evaluate()
    print(results)
    
if __name__ == "__main__":
    config = get_config()
    run_training(config)