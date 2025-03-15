# Code from https://github.com/locuslab/wanda/blob/main/lib/prune.py

import time, pickle, heapq, copy, torch
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .bonsai_utils import *
from .ablate import AblateGPT 
from .layer_merging import *
from .eval import get_layer_output_similarity, eval_ppl_train
# from scipy.sparse.csgraph import connected_components
# from scipy.sparse import csr_matrix

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


def prepare_calibration_input(model, nsamples, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    positional_embeds = []
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            # print(kwargs['position_embeddings'].shape)
            # assert 1==0
            positional_embeds.append(kwargs['position_embeddings'])
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    # positional_embeds = torch.(positional_embeds, dim=0)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, positional_embeds 


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0


#------------- Wanda pruning -----------------#
@torch.no_grad()
def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibration data")
    dataloader, _ = get_loaders(args.calib_dataset,nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, args.nsamples, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs; #!
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings[j])[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings[j])[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


#------------- Shortened-llm pruning -----------------#
def get_block_pruned_network(model_orig, unimportance_order, num_pruned_blocks=1, device=torch.device("cuda:0"), fix_decapoda_config=False, use_bfloat=False, verbose=False):
    # Define the block-pruned architecture with random initialization
    config = copy.deepcopy(model_orig.config)
    print(f"# blocks before pruning: {config.num_hidden_layers}")
    config.__setattr__(
        "num_hidden_layers", (config.num_hidden_layers - num_pruned_blocks)
    )
    print(f"# blocks after pruning: {config.num_hidden_layers}")
    model_pruned = AutoModelForCausalLM.from_config(config)

    # Copy the original model's weights to the pruned model
    model_pruned = copy_weight(
        model_pruned, model_orig, unimportance_order[:num_pruned_blocks], verbose=verbose
    )
    model_pruned = set_model_device_evalmode(
        model_pruned, device, fix_decapoda_config, use_bfloat
    )
    return model_pruned


def copy_weight(model, model_orig, list_pruned_blocks, verbose=False):
    connect_info = {}  # connect_info['TO-small'] = 'FROM-orig'
    connect_info["model.embed_tokens.weight"] = "model.embed_tokens.weight"
    connect_info["model.norm.weight"] = "model.norm.weight"
    connect_info["lm_head.weight"] = "lm_head.weight"

    k = 0
    for k_orig in range(model_orig.config.__getattribute__("num_hidden_layers")):
        if k_orig in list_pruned_blocks:  # uncopied = pruned blocks
            continue
        connect_info[f"model.layers.{k}."] = f"model.layers.{k_orig}."
        print(f"original model.layers.{k_orig} --> pruned model.layers.{k}")
        k = k + 1

    print(f" ** excluded blocks {list_pruned_blocks}")

    t0 = time.perf_counter()
    for k in model.state_dict().keys():
        flag = 0
        k_orig = k
        for prefix_key in connect_info.keys():
            if k.startswith(prefix_key):
                flag = 1
                k_orig = k_orig.replace(prefix_key, connect_info[prefix_key])
                break
        if flag == 1:
            if verbose == True: print(f"** forced COPY {k_orig} -> {k}")
            model.state_dict()[k].copy_(model_orig.state_dict()[k_orig])
    # print(f"copy time --- {(time.perf_counter()-t0):.1f} sec")

    return model


def set_model_device_evalmode(model, device, fix_decapoda_config=False, use_bfloat=True):
    if "cuda" in str(device):
        model.half()
        model = model.to(device)

    if fix_decapoda_config:
        # unwind broken decapoda-research config
        model.config.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
    model.eval()

    if use_bfloat:
        model = model.bfloat16()

    model.seqlen = model.config.max_position_embeddings 
    model.seqlen = min(4096, model.seqlen)
    gc.collect()
    torch.cuda.empty_cache()

    return model


@torch.no_grad()
def prune_shortened_llm(args, model, tokenizer, device=torch.device("cuda:0"), start_layer=1, end_layer=30):
    sorted_ppl_results_path = os.path.abspath(os.path.join(args.save_model, '..', f'sorted_ppl_{args.calib_dataset}.pkl'))
    print(sorted_ppl_results_path)
    # assert 1==0
    if os.path.isfile(sorted_ppl_results_path):
        with open(sorted_ppl_results_path, 'rb') as f:
            sorted_layer_ppl = pickle.load(f)
    else:
        print("loading calibration data")
        dataloader, _ = get_loaders(args.calib_dataset,nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
        ppl = eval_ppl_train(model, dataloader, bs=1, device=device)
        print(f"original ppl {ppl}\n")
        layers = model.model.layers
        unsorted_results = []
        for block_idx in range(len(layers)):
            model_pruned = get_block_pruned_network(
                model,
                unimportance_order=[block_idx],
                num_pruned_blocks=1,
                device=device,
                verbose=False
            )
            # if args.calib_dataset == "bookcorpus": model_pruned.seqlen = 128
            ppl = eval_ppl_train(model_pruned, dataloader, bs=1, device=device)
            print(f"block {block_idx} ppl {ppl}\n")
            unsorted_results.append((block_idx, ppl))
            del model_pruned
            gc.collect()
            torch.cuda.empty_cache()
        sorted_layer_ppl = sorted(unsorted_results, key=lambda x: x[1], reverse=False)
        with open(sorted_ppl_results_path, 'wb') as f:
            pickle.dump(sorted_layer_ppl, f)
    print(sorted_layer_ppl)

    unimportance_order = [x[0] for x in sorted_layer_ppl]
    keep_block_info = [i for i in range(start_layer)] + [i for i in range(end_layer + 1, len(model.model.layers))]
    unimportance_order = [idx for idx in unimportance_order if idx not in keep_block_info]

    # Block-level pruning
    model = get_block_pruned_network(
        model,
        unimportance_order=unimportance_order,
        num_pruned_blocks=end_layer - start_layer + 1 - args.num_components,
        device=device,
        verbose=False
    )
    torch.cuda.empty_cache()
    return model


#------------- SparseGPT pruning -----------------#
@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders(args.calib_dataset,nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    # dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros(
    #     (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    # )
    # cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    # class Catcher(nn.Module):
    #     def __init__(self, module):
    #         super().__init__()
    #         self.module = module
    #     def forward(self, inp, **kwargs):
    #         inps[cache['i']] = inp
    #         cache['i'] += 1
    #         cache['attention_mask'] = kwargs['attention_mask']
    #         cache['position_ids'] = kwargs['position_ids']
    #         raise ValueError
    # layers[0] = Catcher(layers[0])
    # for batch in dataloader:
    #     try:
    #         model(batch[0].to(dev))
    #     except ValueError:
    #         pass
    # layers[0] = layers[0].module
    # torch.cuda.empty_cache()

    # outs = torch.zeros_like(inps)
    # attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']

    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, args.nsamples, dataloader, dev)
    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:  #! multi-gpu
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings[j])[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings[j])[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_bonsai(args, model, tokenizer, device=torch.device("cuda:0")):
    # TODO
    # migrating main() in the Bonsai code.

    str_of_args = args_to_str(args)
    args.save = os.path.join(args.save, str_of_args)
    os.makedirs(args.save, exist_ok=True)
    # wandb_run = wandb.init(
	# 	project=args.wandb_project_name,
	# 	name=str_of_args,
	# 	config=args_to_dict(args),
	# )

    # orig_train_ppl, orig_test_ppl = eval_ppl(model, tokenizer, model.device, dataset=args.calib_dataset)
    # print(f"original train ppl {orig_train_ppl}; original test ppl {orig_test_ppl}")
    
    original_param_count = get_param_count(model)
    model.original_param_count = original_param_count
    cur_sparsity = 1.0 - (get_param_count(model) / original_param_count)
    epoch_ = 1

    while args.sparsity_ratio - cur_sparsity >= args.tol:
       
       # Need to check if we have to clip the sparsity ratio (if the current ratio causes us to overshoot)
        if (cur_sparsity + args.prune_frac) > args.sparsity_ratio:
            # We would overshoot in this case which is not idea.
            old_prune_frac = args.prune_frac
            args.prune_frac = abs(args.sparsity_ratio - cur_sparsity)
            print('We have updated the prune fraction {:.3f} -> {:.3f} to avoid overshooting'.format(old_prune_frac, args.prune_frac))
        
        print('Gathering statistics for pruning')
        save_loc = os.path.join(args.save, 'mask_info_{}.pkl'.format(epoch_))
        if os.path.exists(save_loc):
            print('Successfully loaded past pruning info')
            with open(save_loc, 'rb') as handle:
                mask_info = pkl.load(handle)
        else:
            mask_info = investigate_score_based_mask(args, model, wandb_run=None, epoch_=epoch_)
            with open(save_loc, 'wb') as handle:
                pkl.dump(mask_info, handle)
        
        print('Prune model')
        prune_model(args, model, mask_info, tokenizer) # Do some stuffs here :)
        cur_sparsity = 1.0 - (get_param_count(model) / original_param_count)
		# print(model)

		# Evaluate the performance of the pruned model
        # ppl_train, ppl_test = eval_ppl(model, tokenizer, seed=args.seed, nsamples=args.nsamples, device=model.device, dataset=args.calib_dataset)

        # wandb_run.log({'Sparsity': cur_sparsity, 'TrainPPL': ppl_train, 'TestPPL': ppl_test})
        # print('Sparsity = {:.3f}| Train PPL = {:.3f} | Test PPL = {:.3f}'.format(cur_sparsity, ppl_train, ppl_test))
        print('Sparsity = {:.3f}'.format(cur_sparsity))

        epoch_ += 1

    # wandb_run.log({'sparsity': cur_sparsity})
    print(f'sparsity = {cur_sparsity:.4f}')


#! our method
def find_blocks_indices(sim, matrix):
    n = matrix.shape[0]
    blocks = []
    start = 0

    if n == 1:
        return [(0, 0)]
    while start < n:
        if matrix[start, start] == 1:
            # Find how far the block extends
            end = start + 1
            while end < n:
                # Check if the submatrix is all ones
                if not np.all(matrix[start:end+1, start:end+1] == 1):
                    break
                end += 1

            min_sim = np.min(sim[start:end+1, start:end+1])
            blocks.append(((start, end-1), min_sim))
            start += 1
        else:
            start += 1
    largest_block = sorted(blocks, key=lambda x: (x[0][1]-x[0][0], x[1]), reverse=True)[0][0]

    matrix_before = matrix[:largest_block[0], :largest_block[0]]
    matrix_after = matrix[largest_block[1]+1:, largest_block[1]+1:]

    if matrix_before.shape[0] == 0 and matrix_after.shape[0] == 0:
        ret = [largest_block]
    elif matrix_before.shape[0] == 0 and matrix_after.shape[0] != 0:
        a =  [(x[0] + (largest_block[1] + 1), x[1] + (largest_block[1] + 1)) for x in find_blocks_indices(sim, matrix_after)]
        ret = [largest_block] + a
    elif matrix_before.shape[0] != 0  and matrix_after.shape[0] == 0:
        ret = find_blocks_indices(sim, matrix_before) + [largest_block]
    else:

        a =  [(x[0] + (largest_block[1] + 1), x[1] + (largest_block[1] + 1)) for x in find_blocks_indices(sim, matrix_after)]
        ret = find_blocks_indices(sim, matrix_before) + [largest_block] + a
    
    return ret


def find_blocks_indices_old(matrix):
    n = matrix.shape[0]
    blocks = []
    start = 0
    
    while start < n:
        if matrix[start, start] == 1:
            # Find how far the block extends
            end = start + 1
            while end < n:
                # Check if the submatrix is all ones
                if not np.all(matrix[start:end+1, start:end+1] == 1):
                    break
                end += 1
            blocks.append((start, end-1))
            start = end
        else:
            start += 1
    return blocks


def get_model_size_and_memory(model):
    # output model size and memory usage (in MB)
    model_size = sum(p.numel() for p in model.parameters())
    
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_memory = (param_size + buffer_size) / 1024**2
    return model_size, model_memory


def merge_layers(args, model, tokenizer, device=torch.device("cuda:0")):
    if args.merge_ranges == 'auto':
        #TODO determine the merge ranges automatically
        layer_similarity = get_layer_output_similarity(model, tokenizer, device, dataset=args.calib_dataset, nsamples=args.nsamples)
        binarized_similarity = (layer_similarity > args.merge_thresh).astype(float)
        connected_layers = find_blocks_indices(layer_similarity, binarized_similarity)

         # print(layer_similarity[:5, :5])
        # binarized_similarity_csr = csr_matrix(binarized_similarity)
        # _, connected_layers = connected_components(csgraph=binarized_similarity_csr, directed=False, return_labels=True)
        # np.save("layer_similarity.npy", layer_similarity)


        # connected_layers_old = []
        # old_compression_ratio = 0
        # # model = model.to('cpu')
        # thresh_compression_ratio_dict = {}
        # model = model.to('cpu')
        # torch.cuda.empty_cache()
        # for merge_thresh in np.arange(0.5, 1, 0.0125):
        #     model_copy = copy.deepcopy(model)
        #     binarized_similarity = (layer_similarity > merge_thresh).astype(float)
       
        #     connected_layers = find_blocks_indices(layer_similarity, binarized_similarity)
        #     if connected_layers == connected_layers_old: continue
        #     connected_layers_old = connected_layers
        #     merge_ranges = [x for x in connected_layers if x[1] - x[0] >= args.min_block_size]
            
        #     _, compression_ratio = merge_from_ranges(args, model_copy, merge_ranges)
        #     print(f'threshold: {merge_thresh}, compression ratio: {compression_ratio}')
        #     thresh_compression_ratio_dict[merge_thresh] = (compression_ratio, merge_ranges)
        #     # if compression_ratio >= args.target_compression_ratio and old_compression_ratio < args.target_compression_ratio:
        #     #     break
        #     old_compression_ratio = compression_ratio
        # pickle.dump(thresh_compression_ratio_dict, open(f"thresh_compression_ratio_dict_{args.target_module_name}_{args.target_var}.pkl", "wb"))
        
        # var_compression_ratio_dict = {}
        # binarized_similarity = (layer_similarity > args.merge_thresh).astype(float)
        # connected_layers = find_blocks_indices(layer_similarity, binarized_similarity)
        # merge_ranges = [x for x in connected_layers if x[1] - x[0] >= args.min_block_size]
        # for target_var in [25, 50, 75]:
        #     args.target_var = target_var
        #     # if connected_layers == connected_layers_old: continu
        #     model_copy = copy.deepcopy(model)
        #     _, compression_ratio = merge_from_ranges(args, model_copy, merge_ranges)
        #     print(f'target_var: {target_var}, compression ratio: {compression_ratio}')
        #     var_compression_ratio_dict[target_var] = (compression_ratio, merge_ranges)
        # pickle.dump(var_compression_ratio_dict, open("var_compression_ratio_dict.pkl", "wb"))

        # assert 1==0
        print(connected_layers)
        if not args.target_var: args.min_block_size = args.num_components
        merge_ranges = [x for x in connected_layers if x[1] - x[0] >= args.min_block_size]
        # eval_ppl(args, model, tokenizer, device)
        # assert 1==0
    else:
        merge_ranges = [tuple(map(int, r.split('-'))) for r in args.merge_ranges]

    print(f"Merging layer ranges: {merge_ranges}")
    model, compression_ratio = merge_from_ranges(args, model, merge_ranges)
    # assert 1==0
    
    return model    


def merge_from_ranges(args, model, merge_ranges):
    original_size, original_memory = get_model_size_and_memory(model)
    model, num_components_list = merge_multiple_ranges(model, merge_ranges, args.model_type, args.num_components, args.target_module_name, args.target_var)
    print(f'num_components_list: {num_components_list}')
    print("Updating model configuration after merging...")
    model = update_model_after_merge(model, args.model_type, merge_ranges, num_components_list)
    compressed_size, compressed_memory = get_model_size_and_memory(model)
    return model, compressed_size / original_size