import torch
import torch.nn as nn
import wandb
from transformers import OPTForCausalLM
from sparselinear.transform_util import *
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import TrainingArguments, Trainer, AutoTokenizer
from datasets import load_dataset
import pathlib
import accelerate
import argparse
from peft import LoraConfig, get_peft_model, LoHaConfig, LoHaModel, LoKrConfig, LoKrModel

accelerate.utils.set_seed(1337)

masked_model_path = 'saved_models/opt-125m/prune2-4'
model_name = 'facebook/opt-125m'
fine_tune_dataset = 'wikitext'
exclude_layers = {'lm_head'}

os.environ["WANDB_PROJECT"] = f"reconnect-finetune"

def get_dataset(dataset, tokenizer, seqlen, col_name):
    def tokenizer_func(examples):
        return tokenizer(examples[col_name])
    tokenized_datasets = dataset.map(tokenizer_func, num_proc=8, remove_columns=[col_name])
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // seqlen) * seqlen
        result = {
            k: [t[i : i + seqlen] for i in range(0, total_length, seqlen)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=8,
    )
    return lm_datasets

def get_opt(model):
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

def get_opt_transformed(
        model_path,
        masked=True,
        reconnect_mode=None,
        freeze_weights=True,
        reconnect_factor_e=1,
        reconnect_factor_d=4,
        block_size=8,
        n_groups=-1,
        n_blocks=-1,
        shuffle=False
        ):
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    if masked:
        masks_dict = torch.load(f'{model_path}/masks.pt', map_location=model.device, weights_only=True)
    else:
        masks_dict = None
    group_size = block_size * reconnect_factor_d // reconnect_factor_e
    if n_blocks > 0:
        block_size = -n_blocks
        group_size = -1
    if n_groups > 0:
        group_size = -n_groups
    
    reconnect_modes = {k: reconnect_mode for k in masks_dict.keys()}
    
    replace_all_nnLinear(
        model, masks_dict,
        reconnect_modes=reconnect_modes,
        reconnect_block_size=block_size,
        group_size=group_size,
        shuffle=shuffle,
        freeze_weights=freeze_weights
        )
    return model

def get_module(model: nn.Module, module_name: str):
    tokens = module_name.strip().split('.')
    parent = model
    for t in tokens:
        if not t.isnumeric():
            parent = getattr(parent, t)
        else:
            parent = parent[int(t)]
    return parent

def get_opt_lora(model_path, scaling_factor=0.25):
    def get_lora_rank(layer):
        return int((layer.in_features * layer.out_features * scaling_factor)/(layer.in_features + layer.out_features))
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    masks_dict = torch.load(f'{model_path}/masks.pt', map_location=model.device, weights_only=True)
    rank_dict = {}
    for k, m in masks_dict.items():
        ratio = m.sum().item() / m.numel()
        layer = get_module(model, k)
        rank = get_lora_rank(layer)
        rank_dict[k] = rank
    config = LoraConfig(
        rank_pattern=rank_dict,
        alpha_pattern=rank_dict,
        use_rslora=True,
        init_lora_weights="gaussian",
        target_modules=list(rank_dict.keys())
    )
    model = get_peft_model(model, config)
    return model

def get_opt_loha(model_path, scaling_factor=0.25):
    def get_loha_rank(layer):
        return int((layer.in_features * layer.out_features * scaling_factor)/(layer.in_features + layer.out_features) / 2)
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    masks_dict = torch.load(f'{model_path}/masks.pt', map_location=model.device, weights_only=True)
    rank_dict = {}
    for k, m in masks_dict.items():
        ratio = m.sum().item() / m.numel()
        layer = get_module(model, k)
        rank = get_loha_rank(layer)
        rank_dict[k] = rank
    config = LoHaConfig(
        rank_pattern=rank_dict,
        alpha_pattern=rank_dict,
        target_modules=list(rank_dict.keys())
    )
    model = LoHaModel(model, config, "default")
    return model

def get_opt_lokr(model_path, scaling_factor=0.25, decompose_factor=8):
    def get_lokr_rank(layer):
        p = layer.in_features
        q = layer.out_features
        f = decompose_factor
        return int(((scaling_factor * p * q - f * f) * f) / (p + q))
    model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    masks_dict = torch.load(f'{model_path}/masks.pt', map_location=model.device, weights_only=True)
    rank_dict = {}
    for k, m in masks_dict.items():
        ratio = m.sum().item() / m.numel()
        layer = get_module(model, k)
        rank = get_lokr_rank(layer)
        rank_dict[k] = rank
    config = LoKrConfig(
        # decompose_factor=decompose_factor,
        # decompose_both=True,
        rank_pattern=rank_dict,
        alpha_pattern=rank_dict,
        target_modules=list(rank_dict.keys())
    )
    model = LoKrModel(model, config, "default")
    return model
    

# class TrainArgs:
#     output_dir = masked_model_path + "-finetune"
#     gradient_accumulation_steps = 5
#     max_grad_norm = 1.0
#     batch_size = 12
#     learning_rate = 5e-5
#     max_iters = 3000
#     warmup_steps = 0.1
#     min_lr = 5e-6
#     ddp_backend = 'nccl'
#     dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
#     torch_compile = True
    
def main(mode, dataset, factor_e, factor_d, block_size, n_groups, n_blocks, shuffle, lr_scale):
    lr = 5e-5
    # args = TrainArgs()
    if mode == 'dense':
        train_name = 'dense_baseline'
    elif mode == 'sparse':
        train_name = '24_finetune'
    elif mode == 'sparse_regrow':
        train_name = '24_regrow'
    elif mode in ['reconnect', 'oft', 'oft_approx']:
        train_name = f'24_{mode}'
        if n_groups < 0 and n_blocks < 0:
            train_name += f'_{factor_e}_{factor_d}'
        if not shuffle:
            train_name += '_noshuffle'
        # lr = 5e-4
    elif mode == 'lora':
        train_name = f'24_lora_{factor_e}_{factor_d}'
        lora_scaling = factor_e / factor_d
    elif mode == 'loha':
        train_name = f'24_loha_{factor_e}_{factor_d}'
        lora_scaling = factor_e / factor_d
    else:
        raise ValueError("mode must be 'dense', 'sparse', 'reconnect', 'oft', 'oft_approx', 'lora' or 'loha'")
    
    train_name = f"{train_name}_{dataset}"
    training_args = TrainingArguments(
        # output_dir = masked_model_path + "-dense_baseline",
        output_dir = masked_model_path + f"-{train_name}",
        overwrite_output_dir=True,
        eval_strategy='steps',
        prediction_loss_only=True,
        gradient_accumulation_steps=5,
        max_grad_norm=1.0,
        max_steps=3000,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        logging_strategy='steps',
        logging_steps=200,
        save_strategy='steps',
        save_steps=200,
        save_total_limit=3,
        # jit_mode_eval=True,
        bf16=True,
        ddp_backend='nccl',
        load_best_model_at_end=True,
        optim='adamw_torch_fused',
        report_to=None,
        accelerator_config={
            "split_batches": True,
        },
        logging_first_step=True,
        # dataloader_prefetch_factor=2,
        auto_find_batch_size=True,
        torch_compile=True,
        eval_on_start=True,
        learning_rate=lr,
    )
    
    if mode == 'dense':
        model = get_opt(model_name)
    elif mode == 'sparse':
        model = get_opt_transformed(masked_model_path, freeze_weights=False)
    elif mode == 'sparse_regrow':
        model = get_opt_transformed(masked_model_path, masked=False, freeze_weights=False)
    elif mode in ['reconnect', 'oft', 'oft_approx']:
        model = get_opt_transformed(
            masked_model_path, reconnect_mode=mode,
            reconnect_factor_e=factor_e, reconnect_factor_d=factor_d,
            block_size=block_size, n_groups=n_groups, n_blocks=n_blocks, shuffle=shuffle)
        training_args.learning_rate *= lr_scale
    elif mode == 'lora':
        model = get_opt_lora(masked_model_path, scaling_factor=lora_scaling)
    elif mode == 'loha':
        model = get_opt_loha(masked_model_path, scaling_factor=lora_scaling)
    else:
        raise ValueError("invalide mode")
    
    dataset_dict = {
        'wikitext': ('wikitext', 'wikitext-2-raw-v1', 'text'),
        'ptb': ('ptb_text_only', 'penn_treebank', 'sentence')
    }
    
    dataset_id = dataset_dict[dataset]
    dataset = load_dataset(dataset_id[0], dataset_id[1])
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm_datasets = get_dataset(dataset, tokenizer, model.seqlen, dataset_id[2])
    training_args.training_args = {
        'model_name': model_name,
        'dataset_name': dataset_id[1],
        'train_name': train_name,
        'mode': mode,
        'param_factor_e': factor_e,
        'param_factor_d': factor_d,
        'n_groups': n_groups,
        'shuffle': shuffle
    }
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset=lm_datasets['train'],
        eval_dataset=lm_datasets['validation']
    )
    trainer.train()

class CustomAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        if "block" in self.dest:
            setattr(namespace, "block_given", True)
        elif "factor" in self.dest:
            setattr(namespace, "factor_given", True)
        elif "group" in self.dest:
            setattr(namespace, "group_given", True)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('mode', type=str, choices=['dense', 'sparse', 'sparse_regrow', 'reconnect', 'lora', 'oft', 'oft_approx', 'loha'])
    argparser.add_argument('dataset', type=str, choices=['wikitext', 'ptb'])
    argparser.add_argument('--param_factor_e', '-pm', type=int, default=1, action=CustomAction)
    argparser.add_argument('--param_factor_d', '-pd', type=int, default=4, action=CustomAction)
    argparser.add_argument('--block_size', '-bs', type=int, default=8)
    argparser.add_argument('--n_groups', '-ng', type=int, default=-1)
    argparser.add_argument('--n_blocks', '-nb', type=int, default=-1)
    argparser.add_argument('--shuffle', '-s', action='store_true')
    argparser.add_argument('--lr_scale', '-ls', type=float, default=10.0)
    args = argparser.parse_args()
    
    
    main(
        args.mode, args.dataset,
        args.param_factor_e, args.param_factor_d,
        args.block_size, args.n_groups, args.n_blocks,
        args.shuffle, args.lr_scale
        )