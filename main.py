
import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from utils.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_bonsai, prune_ablate, check_sparsity, find_layers
from utils.eval import eval_ppl, eval_zero_shot
# bonsai
from utils.lib.modelling_llama_mod import LlamaForCausalLM

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def get_llm_bonsai(model_name, cache_dir="llm_weights"):
	model = LlamaForCausalLM.from_pretrained(
		model_name, 
		torch_dtype=torch.float16, 
		cache_dir=cache_dir, 
		low_cpu_mem_usage=True, 
		device_map="auto"
	)
	model.seqlen = model.config.max_position_embeddings 
	if ('13b' in model_name) or ('65b' in model_name):
		model.seqlen = 2048 #Based on the values from the Lora-prune paper
	return model

def get_nonzero_param_count(model, exclude=['embed', 'head']):
    #! unified sparsity calculation
    return sum([
        (p != 0).sum().item() 
        for n, p in model.named_parameters() 
        if not any(x in n for x in exclude)
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-2-7b-hf', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default='unstructured', choices=["unstructured", "4:8", "2:4"])  # not for bonsai
    parser.add_argument('--calib_dataset', type=str, default="c4", choices=["wikitext2", "c4"])
    parser.add_argument("--prune_method", default='wanda', type=str, choices=["magnitude", "wanda", "sparsegpt", "bonsai",
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    #! --- for Bonsai starts
    parser.add_argument('--bonsai_prune_method', type=str, default="wanda", choices=["magnitude", "wanda", "random"])
    parser.add_argument('--tol', type=float, default=0.02, help="What level of tolerance close to the target sparsity to accept")
    parser.add_argument('--prune_frac', type=float, default=0.05, help='Fraction of weights to prune at a time')
    parser.add_argument('--masks_per_iter', type=int, default=200, help='How many masks to generate per-iteration')
    parser.add_argument('--bsz', type=int, default=1, help='Instantaneous batch size for forward pass')
    parser.add_argument('--mlp_attn_ratio', type=float, default=1.0, help="For a given prune_frac, the ratio of the pruning for attn vrs mlp")
    parser.add_argument('--no_perturb', action="store_true", help="We do not perform any perturbation")
    # Hyperparams for scoring model
    parser.add_argument('--sm_reg_weight', type=str, default='[1e2, 1e-4, 0]', help='reg-weight to use')
    parser.add_argument('--sm_lr_factor', type=str, default='[100, 10, 1, 0.1]', help='lr factor to use for fitting linear model')
    parser.add_argument('--sm_reg_type', type=str, default="l1", help='type of regularization to apply')
    parser.add_argument('--sm_lin_model_type', type=str, default="global", help='type of regularization to apply') 
    parser.add_argument('--sm_bsz', type=str, default='[32, 64, 128]', help='batch size for fitting linear model')
    parser.add_argument('--sm_nepochs', type=int, default=50, help='number of epochs to use to fit the linear model')
    #! --- for Bonsai ends
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default='output', help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    if not args.prune_method == 'bonsai':
        model = get_llm(args.model, args.cache_dir)
    else:
        model = get_llm_bonsai(args.model, args.cache_dir)
    model.eval()
    original_nozero_param_count = get_nonzero_param_count(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "bonsai":
            prune_bonsai(args, model, tokenizer, device)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    # ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    cur_nozero_param_count = get_nonzero_param_count(model)
    sparsity_ratio_unified = 1 - cur_nozero_param_count / original_nozero_param_count
    print(f"sparsity unified {sparsity_ratio_unified:.4f}")
    print("*"*30)
    # ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    # if not os.path.exists(args.save):
    #     os.makedirs(args.save)
    # save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    # with open(save_filepath, "w") as f:
    #     print("method\tactual_sparsity\tppl_test", file=f, flush=True)
    #     print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ['openbookqa', 'winogrande']#["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]#["leaderboard"]#, 
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()
