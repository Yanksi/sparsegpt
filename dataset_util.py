from datasets import load_dataset

def get_wikitext():
    return load_dataset('wikitext', 'wikitext-2-raw-v1')

def get_ptb():
    ds = load_dataset('ptb_text_only', 'penn_treebank')
    ds = ds.rename_column('sentence', 'text')
    return ds

def get_openwebtext():
    ds = load_dataset('openwebtext')
    split_datasets = ds['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=False)
    split_datasets['validation'] = split_datasets.pop('test')
    return split_datasets

def process_dataset(ds, tokenizer, seqlen):
    def tokenize_and_group_texts(examples):
        examples = tokenizer(examples["text"])
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // seqlen) * seqlen
        result = {
            k: [t[i : i + seqlen] for i in range(0, total_length, seqlen)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    lm_datasets = ds.map(
        tokenize_and_group_texts,
        batched=True,
        batch_size=1000,
        num_proc=128,
        remove_columns=['text'],
        load_from_cache_file=True
    )
    return lm_datasets

def get_dataset(dataset, tokenizer, seqlen):
    dataset_dict = {
        'wikitext': get_wikitext,
        'ptb': get_ptb,
        'openwebtext': get_openwebtext
    }
    ds = dataset_dict[dataset]()
    return process_dataset(ds, tokenizer, seqlen)