def get_config():
    return {
        "model_name" : "google-t5/t5-small",
        "datasource": "findnitai/english-to-hinglish",
        "src_lang" : "en",
        "tgt_lang" : "hi_ng",
        "output_dir" : "results/",
        "eval_strategy" : "steps",
        "eval_steps" : 0.1,
        "lr" : 1e-5,
        "batch_size" : 24,
        "epochs" : 3,
        "saved_model_path" : "/Users/arnavdixit/Desktop/Personal/LLM Practice/Finetuning-Language-Translation/results/checkpoint-21276",
        "custom_tokenizer_path" : "custom_tokenizer.json"
    }