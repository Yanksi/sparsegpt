import time

import torch
import torch.nn as nn

from quant import *
from sparsegpt import *
from modelutils import *

from sparselinear.util import *
import pathlib

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False 


def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def opt_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model = model.to(dev)
    linear_layers = get_layers_with_type(layers, nn.Linear, prefix='model.decoder.layers.')
    linear_to_names = get_layer_to_names(linear_layers)

    sparse_gpts = {}

    for l in linear_to_names:
        sparse_gpts[l] = SparseGPT(l)
    
    def add_batch_module(module, inp, out):
        if module in sparse_gpts:
            sparse_gpts[module].add_batch(inp[0].data, out.data)
    
    handle = torch.nn.modules.module.register_module_forward_hook(add_batch_module)

    print("Collecting calibration data ...")
    for batch in dataloader:
        model(batch[0].to(dev))
    handle.remove()

    masks = {}
    print("Pruning ...")
    for l, s in sparse_gpts.items():
        print(linear_to_names[l][0] if len(linear_to_names[l]) == 1 else linear_to_names[l])
        sparsity = args.sparsity
        pruning_mask = s.fasterprune(
            sparsity, prunen=args.prunen, prunem=args.prunem, percdamp=args.percdamp, blocksize=args.blocksize
        )
        s.free()
        masks[linear_to_names[l][0]] = pruning_mask

    model.config.use_cache = use_cache
    return masks

@torch.no_grad()
def opt_gmp(model, target_sparsity):
    print("Starting ...")
    layers = model.model.decoder.layers
    linear_layers = get_layers_with_type(layers, nn.Linear, prefix='model.decoder.layers.')
    masks = {}
    for n, l in linear_layers.items():
        W = l.weight
        thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * target_sparsity)]
        masks[n] = torch.abs(W) > thresh
        W.data[torch.abs(W.data) <= thresh] = 0
    return masks

@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    model = model.to(dev)
    use_cache = model.config.use_cache
    model.config.use_cache = False

    testenc = testenc[:, :(nsamples * model.seqlen)].contiguous().reshape(nsamples, model.seqlen)
    losses = []
    batch_size = 16
    for i in range(0, nsamples, batch_size):
        batch = testenc[i:min(i + batch_size, nsamples)].to(dev)
        losses.append(model(batch, labels=batch, return_dict=True).loss)
    ppl = torch.exp(torch.stack(losses).mean())
    model.config.use_cache = use_cache
    return ppl

@torch.no_grad()
def opt_eval_all(model, tag):
    print(tag)
    perplexities = {}
    for dataset in ['wikitext2', 'ptb', 'c4']:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        perplexity = opt_eval(model, testloader, DEV)
        print(f"Perplexity: {perplexity.item():3f}")
        perplexities[dataset] = perplexity.item()
    return perplexities


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str, 
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--blocksize', type=int, default=128,
        help='Blocksize to use for adaptive mask selection.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16,
        help='Whether to quantize as well.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
       '--invert', action='store_true', 
       help='Invert subset.'
    )
    parser.add_argument(
       '--save', type=str, default='',
       help='Path to saved model.'
    )
    parser.add_argument(
       '--log_wandb', action='store_true',
       help='Whether to log to wandb.'
    )
    parser.add_argument(
        '--baseline', action='store_true',
        help='Whether to test the dense baseline.'
    )

    args = parser.parse_args()

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(project="prune", name=args.model.split('/')[1], config=args)

    model = get_opt(args.model)
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    print(f"Model: {args.model}")
    perplexities = {}
    if args.baseline:
        perplexities['dense'] = opt_eval_all(model, "dense")

    if (args.sparsity or args.prunen):
        model = get_opt(args.model)
        model.eval()
        # tick = time.time()
        masks = opt_sequential(model, dataloader, DEV)
        if args.save:
            model.save_pretrained(args.save)
            torch.save(masks, f'{args.save}/masks.pt')
        perplexities['SparseGPT'] = opt_eval_all(model, "SparseGPT")
        
    if args.gmp and (args.sparsity or args.prunen):
        if args.prunen:
            target_sparsity = args.prunen / args.prunem
        else:
            target_sparsity = args.sparsity
        model = get_opt(args.model)
        model.eval()
        masks = opt_gmp(model, target_sparsity)
        if args.save:
            save_path = pathlib.Path(args.save).parent / f"gmp_{int(target_sparsity * 100):02d}"
            model.save_pretrained(save_path)
            torch.save(masks, f'{save_path}/masks.pt')
        perplexities['GMP'] = opt_eval_all(model, "GMP")
    
    if args.log_wandb:
        data = []
        for method, vals in perplexities.items():
            for dataset, v in vals.items():
                data.append([dataset, method, v])

        table = wandb.Table(data=data, columns = ["dataset", "method", "perplexity"])
        wandb.log({"Grouped_metrics" : table}) 
        wandb.run.finish()
