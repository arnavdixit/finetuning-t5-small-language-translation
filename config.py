def get_config():
    return {
        "model_name" : "google-t5/t5-small",
        "datasource": "findnitai/english-to-hinglish",
        "src_lang" : "en",
        "tgt_lang" : "hi_ng",
        "output_dir" : "./results",
        "lr" : 1e-5,
        "batch_size" : 16,
        "epochs" : 3
    }