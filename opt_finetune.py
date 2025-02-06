import torch
import torch.nn as nn
import wandb
from transformers import OPTForCausalLM, LlamaForCausalLM
from sparselinear.transform_util import *
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pathlib
import accelerate
import argparse
from peft import LoraConfig, get_peft_model, LoHaConfig, LoHaModel, LoKrConfig, LoKrModel
import optuna
from dataset_util import get_dataset

accelerate.utils.set_seed(1337)

# masked_model_path = 'saved_models/opt-125m/prune2-4'
# model_name = 'facebook/opt-125m'

models = {
    "opt-125m": {
        "name": "facebook/opt-125m",
        "masked_path": "saved_models/opt-125m/prune2-4"
    },
    "llama-1b": {
        "name": "meta-llama/Llama-3.2-1B",
        "masked_path": "saved_models/llama-1B/prune2-4"
    }
}
sequence_length = 2048
fine_tune_dataset = 'wikitext'
exclude_layers = {'lm_head'}


def get_model(model):
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = sequence_length
    return model

def get_model_transformed(
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
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto')
    model.seqlen = sequence_length
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

def get_model_lora(model_path, scaling_factor=0.25):
    def get_lora_rank(layer):
        return int((layer.in_features * layer.out_features * scaling_factor)/(layer.in_features + layer.out_features))
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype='auto')
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

# def get_opt_loha(model_path, scaling_factor=0.25):
#     def get_loha_rank(layer):
#         return int((layer.in_features * layer.out_features * scaling_factor)/(layer.in_features + layer.out_features) / 2)
#     model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
#     model.seqlen = model.config.max_position_embeddings
#     masks_dict = torch.load(f'{model_path}/masks.pt', map_location=model.device, weights_only=True)
#     rank_dict = {}
#     for k, m in masks_dict.items():
#         ratio = m.sum().item() / m.numel()
#         layer = get_module(model, k)
#         rank = get_loha_rank(layer)
#         rank_dict[k] = rank
#     config = LoHaConfig(
#         rank_pattern=rank_dict,
#         alpha_pattern=rank_dict,
#         target_modules=list(rank_dict.keys())
#     )
#     model = LoHaModel(model, config, "default")
#     return model

# def get_opt_lokr(model_path, scaling_factor=0.25, decompose_factor=8):
#     def get_lokr_rank(layer):
#         p = layer.in_features
#         q = layer.out_features
#         f = decompose_factor
#         return int(((scaling_factor * p * q - f * f) * f) / (p + q))
#     model = OPTForCausalLM.from_pretrained(model_path, torch_dtype='auto')
#     model.seqlen = model.config.max_position_embeddings
#     masks_dict = torch.load(f'{model_path}/masks.pt', map_location=model.device, weights_only=True)
#     rank_dict = {}
#     for k, m in masks_dict.items():
#         ratio = m.sum().item() / m.numel()
#         layer = get_module(model, k)
#         rank = get_lokr_rank(layer)
#         rank_dict[k] = rank
#     config = LoKrConfig(
#         # decompose_factor=decompose_factor,
#         # decompose_both=True,
#         rank_pattern=rank_dict,
#         alpha_pattern=rank_dict,
#         target_modules=list(rank_dict.keys())
#     )
#     model = LoKrModel(model, config, "default")
#     return model

    
def main(mode, dataset, model, factor_e, factor_d, block_size, n_groups, n_blocks, shuffle, lr, lr_scale, hp_finding=False):
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
    
    model_name = models[model]['name']
    masked_model_path = models[model]['masked_path']
    
    # text_dataset = dataset_dict[dataset]()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    lm_datasets = get_dataset(dataset, tokenizer, 2048)
    
    ddp_world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    is_ddp_run = ddp_world_size > 1
    
    train_name = f"{train_name}_{dataset}"
    training_args = TrainingArguments(
        # output_dir = masked_model_path + "-dense_baseline",
        output_dir = masked_model_path + f"-{train_name}",
        overwrite_output_dir=True,
        eval_strategy='steps',
        eval_steps=200,
        prediction_loss_only=True,
        gradient_accumulation_steps=5,
        max_grad_norm=1.0,
        max_steps=3000,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        logging_strategy='steps',
        logging_steps=200,
        save_strategy='steps' if not hp_finding else 'no',
        save_steps=200,
        save_total_limit=3,
        # jit_mode_eval=True,
        bf16=True,
        ddp_backend='nccl' if is_ddp_run else None,
        load_best_model_at_end=True if not hp_finding else False,
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
        get_model = lambda _: get_model(model_name)
    elif mode == 'sparse':
        get_model = lambda _: get_model_transformed(masked_model_path, freeze_weights=False)
    elif mode == 'sparse_regrow':
        get_model = lambda _: get_model_transformed(masked_model_path, masked=False, freeze_weights=False)
    elif mode in ['reconnect', 'oft', 'oft_approx']:
        get_model = lambda _: get_model_transformed(
            masked_model_path, reconnect_mode=mode,
            reconnect_factor_e=factor_e, reconnect_factor_d=factor_d,
            block_size=block_size, n_groups=n_groups, n_blocks=n_blocks, shuffle=shuffle)
        training_args.learning_rate *= lr_scale
    elif mode == 'lora':
        get_model = lambda _: get_model_lora(masked_model_path, scaling_factor=lora_scaling)
    # elif mode == 'loha':
    #     get_model = lambda _: get_opt_loha(masked_model_path, scaling_factor=lora_scaling)
    else:
        raise ValueError("invalide mode")
    
    training_args.training_args = {
        'model_name': model_name,
        'dataset_name': dataset,
        'train_name': train_name,
        'mode': mode,
        'param_factor_e': factor_e,
        'param_factor_d': factor_d,
        'n_groups': n_groups,
        'shuffle': shuffle
    }
    
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        }
    
    def wandb_hp_space(trial):
        return {
            "method": "bayes",
            "metric": {"name": "objective", "goal": "minimize"},
            "parameters": {
                "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 1e-2}
            },
        }
    
    trainer = Trainer(
        model = None,
        model_init = get_model,
        args = training_args,
        train_dataset=lm_datasets['train'],
        eval_dataset=lm_datasets['validation']
    )
    
    os.environ["WANDB_PROJECT"] = 'reconnect-finetune' if not hp_finding else 'reconnect-finetune-lr'
    os.environ["WANDB_RUN_GROUP"] = train_name
    if hp_finding:
        trainer.hyperparameter_search(
            direction='minimize',
            backend='optuna',
            hp_space=optuna_hp_space,
            n_trials=20,
            study_name=train_name,
            storage='sqlite:///db.sqlite3',
            load_if_exists=True
        )
    else:
        trainer.train()
        # pass


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
    argparser.add_argument('dataset', type=str, choices=['wikitext', 'ptb', 'openwebtext'])
    argparser.add_argument('--model', type=str, choices=['opt-125m', 'llama-1b'], default='opt-125m')
    argparser.add_argument('--param_factor_e', '-pm', type=int, default=1, action=CustomAction)
    argparser.add_argument('--param_factor_d', '-pd', type=int, default=4, action=CustomAction)
    argparser.add_argument('--block_size', '-bs', type=int, default=8)
    argparser.add_argument('--n_groups', '-ng', type=int, default=-1)
    argparser.add_argument('--n_blocks', '-nb', type=int, default=-1)
    argparser.add_argument('--shuffle', '-s', action='store_true')
    argparser.add_argument('--lr', '-lr', type=float, default=5e-5)
    argparser.add_argument('--lr_scale', '-ls', type=float, default=30.0)
    argparser.add_argument('--hp_finding', '-hp', action='store_true')
    args = argparser.parse_args()
    
    
    main(
        args.mode, args.dataset, args.model,
        args.param_factor_e, args.param_factor_d,
        args.block_size, args.n_groups, args.n_blocks,
        args.shuffle, args.lr, args.lr_scale, args.hp_finding
        )