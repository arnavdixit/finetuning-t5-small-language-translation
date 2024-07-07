from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer

from tokenizers.implementations import SentencePieceBPETokenizer
from tokenizers import Tokenizer

import sentencepiece as spm

from datasets import load_dataset

import torch
from torch.utils.data import random_split

from config import get_config

from pathlib import Path

def get_ds(config):
    raw_ds = load_dataset(config['datasource'], split = "train")
    return raw_ds

def get_custom_tokenizer(config, train_data):
    tokenizer_path = Path(config["custom_tokenizer_path"])
    if not Path.exists(tokenizer_path):
        all_texts = []
        for item in train_data:
            all_texts.append(item['translation'][config['src_lang']])
            all_texts.append(item['translation'][config['tgt_lang']])
                
        custom_tokenizer = SentencePieceBPETokenizer()
        custom_tokenizer.train_from_iterator(all_texts, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
        custom_tokenizer.save(str(tokenizer_path))
    else:
        custom_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return custom_tokenizer
    

def train_model(config):
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
    
    print("Getting Custom Tokenizer")
    custom_tokenzier = get_custom_tokenizer(config, train_ds)
    new_tokens = [token for token in custom_tokenzier.get_vocab() if token not in tokenizer.get_vocab()]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    print("Model, Data and Tokenizer Ready for training.")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir = config['output_dir'],
        eval_strategy = config['eval_strategy'],
        eval_steps = config['eval_steps'],
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
    train_model(config)