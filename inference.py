from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from torchmetrics.text import BLEUScore, CharErrorRate, WordErrorRate

import torch

from config import get_config

def run_evaluation(config):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = config["saved_model_path"]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    test_txt = "When do you want to watch the movie?"
    
    tokenized_input = tokenizer(test_txt)
    output = model(tokenized_input)
    print(output)
    
    
if __name__ == "__main__":
    config = get_config()
    run_evaluation(config)
    