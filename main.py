import argparse, os, torch, json, pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from importlib.metadata import version

from utils.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_shortened_llm, prune_bonsai, prune_ablate, check_sparsity, merge_layers, select_layers, get_model_size_and_memory, prune_shortgpt, prune_laco
from utils.eval import eval_ppl, eval_zero_shot
# bonsai
from utils.lib.modelling_llama_mod import LlamaForCausalLM
from calflops import calculate_flops

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def load_model_with_rope_scaling_adjustment(model_name, cache_dir='llm_weights', use_auth_token=True, use_bfloat16=False):
    # local_path = get_local_model_path(model_path)
    
    # Create base directory if it doesn't exist
    # os.makedirs(local_path, exist_ok=True)
    
    # Check if model already exists locally
    # if not check_model_exists(local_path):
    #     print(f"Model not found in {local_path}. Downloading...")
    #     try:
    #         # Download model files to local path
    #         config = AutoConfig.from_pretrained(model_path, use_auth_token=use_auth_token)
    #         model = AutoModelForCausalLM.from_pretrained(
    #             model_path,
    #             use_auth_token=use_auth_token,
    #             torch_dtype=torch.bfloat16 if use_bfloat16 else None
    #         )
    #         tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=use_auth_token)
            
    #         # Save files locally
    #         config.save_pretrained(local_path)
    #         model.save_pretrained(local_path)
    #         tokenizer.save_pretrained(local_path)
    #         print(f"Model downloaded and saved to {local_path}")
    #     except Exception as e:
    #         print(f"Error downloading model: {e}")
    #         raise
    # else:
    #     print(f"Loading model from local cache: {local_path}")

    
    # try:
    #     with open(os.path.join(local_path, 'config.json'), 'r') as f:
    #         config_dict = json.load(f)
        
    #     if 'rope_scaling' in config_dict:
    #         original_rope_scaling = config_dict['rope_scaling'].copy()
    #         print(f"Original rope_scaling: {original_rope_scaling}")
            
    #         config_dict['rope_scaling'] = {
    #             'type': original_rope_scaling.get('rope_type', 'linear'),
    #             'factor': original_rope_scaling.get('factor', 1.0)
    #         }
    #         print(f"Adjusted rope_scaling: {config_dict['rope_scaling']}")
        
    #     config = AutoConfig.from_pretrained(None, **config_dict)
        
    #     if use_bfloat16:
    #         model = AutoModelForCausalLM.from_pretrained(
    #             local_path,
    #             config=config,
    #             torch_dtype=torch.bfloat16
    #         )
    #     else:
    #         model = AutoModelForCausalLM.from_pretrained(
    #             local_path,
    #             config=config
    #         )
    # except Exception as e:
    #     print(f"Error loading model with config adjustment: {e}")
    #     print("Attempting to load model without config adjustment...")
    #     if use_bfloat16:
    #         model = AutoModelForCausalLM.from_pretrained(
    #             local_path,
    #             torch_dtype=torch.bfloat16
    #         )
    #     else:
    #         model = AutoModelForCausalLM.from_pretrained(local_path)
    config = AutoConfig.from_pretrained(model_name)
    if hasattr(config, 'rope_scaling'):
        original_rope_scaling = config.rope_scaling
        if original_rope_scaling is not None:
            print(f"Original rope_scaling: {original_rope_scaling}")
            
            config.rope_scaling = {
                'type': original_rope_scaling.get('rope_type', 'linear'),
                'factor': original_rope_scaling.get('factor', 1.0)
            }
        print(f"Adjusted rope_scaling: {config.rope_scaling}")
    # # print(config_dict)
    # if 'rope_scaling' in config_dict and config_dict['rope_scaling'] is not None:
    #     original_rope_scaling = config_dict['rope_scaling'].copy()
    #     print(f"Original rope_scaling: {original_rope_scaling}")
        
    #     config_dict['rope_scaling'] = {
    #         'type': original_rope_scaling.get('rope_type', 'linear'),
    #         'factor': original_rope_scaling.get('factor', 1.0)
    #     }
    #     print(f"Adjusted rope_scaling: {config_dict['rope_scaling']}")
    
    # config = AutoConfig.from_pretrained(model_name, **config_dict)
    # turn config_dict back to config
    # config = Config.from_dict(config_dict)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                cache_dir=cache_dir,
                                                config=config,
                                                torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
                                                low_cpu_mem_usage=True, 
                                                device_map="auto")
    model.seqlen = model.config.max_position_embeddings 
    return model


def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    print(model.seqlen)
    model.seqlen = min(4096, model.seqlen)
    print(model.seqlen)
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
    parser.add_argument('--model', default='meta-llama/llama-3.1-8b', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default='unstructured', choices=["unstructured", "4:8", "2:4"])  # not for bonsai
    parser.add_argument('--calib_dataset', type=str, default="c4", choices=["wikitext2", "c4", "bookcorpus", "random"])
    parser.add_argument("--prune_method", default='merge', type=str, choices=["select", "merge", "magnitude", "wanda", "sparsegpt", "shortened_llm", "bonsai",
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search", "shortgpt", "laco", "none"])
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

    #! --- for merging starts
    parser.add_argument("--model_type", choices=["opt", "llama", "gemma"], default="llama", help="Type of model to use")
    parser.add_argument("--merge_ranges", nargs="+", default='auto', help="Ranges of layers to merge, e.g., '2-12 14-17 18-19', or 'auto'")
    parser.add_argument("--merge_thresh", type=float, default=None, help="Similarity threshold for merging the layers")
    parser.add_argument("--use_bfloat16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--min_block_size", type=int, default=1, help="Minimum size of the blocks to use for merging")
    parser.add_argument("--num_components", type=int, default=1, help="Number of principal components to use for merging")
    parser.add_argument("--target_var", type=int, default=None, help="Target variance for merging")
    parser.add_argument("--target_module_name", type=str, default='input_layernorm',
                        choices=['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.o_proj', 
                                 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'input_layernorm', 'post_attention_layernorm'], 
                        help='which module name to decide number of components')
    parser.add_argument("--target_compression_ratio", type=float, default=0.5, help="target compression_ratio for the model")
    #! --- for merging ends
    
    #! --- for LACO starts
    parser.add_argument("--merge_layers", type=int, default=4, help="Number of layers to merge together in LACO")
    parser.add_argument("--threshold", type=float, default=0.25, help="Similarity threshold for LACO merging")
    #! --- for LACO ends
    
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    # parser.add_argument('--save', type=str, default='output', help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()
    calib_action = next(action for action in parser._actions if action.dest == 'calib_dataset')
    calib_choices = calib_action.choices
    if 'random' in calib_choices:
        calib_choices.remove('random')

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    save_root = f'/home/scratch/changl8/prune_llm/{args.model.split("/")[-1]}'
    if args.save_model:
        os.makedirs(args.save_model, exist_ok=True)

    if not args.save_model:
        if args.prune_method == "merge" or args.prune_method == "select":
            if args.merge_ranges == 'auto':
                if args.target_var:
                    args.save_model = f'{save_root}/{args.prune_method}_{args.target_var}_{args.merge_thresh}'
                else:
                    args.save_model = f'{save_root}/{args.prune_method}_{args.num_components}_{args.merge_thresh}'
            else:
                if args.target_var:
                    args.save_model = f'{save_root}/{args.prune_method}_{args.target_var}_{args.merge_ranges}'
                else:
                    args.save_model = f'{save_root}/{args.prune_method}_{args.num_components}_{args.merge_ranges}'
        elif args.prune_method in ["shortened_llm", "shortgpt", "laco"]:
            args.save_model = f'{save_root}/{args.prune_method}/{args.num_components}'
        elif args.prune_method == 'none':
            args.save_model = f'{save_root}/{args.prune_method}'
        else: args.save_model = f'{save_root}/{args.prune_method}_{args.sparsity_ratio}'

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")

    # if args.prune_method == 'merge':
    #     model = load_model_with_rope_scaling_adjustment(args.model, args.cache_dir, use_bfloat16=args.use_bfloat16)
        # model = get_llm(args.model, args.cache_dir)
        # assert 1==0
        # model = AutoModelForCausalLM.from_pretrained(args.save_model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
        
    model = get_llm(args.model, args.cache_dir)
    if args.calib_dataset == "bookcorpus": model.seqlen = 128
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # ppl_train,ppl_test = eval_ppl(model, tokenizer, seed=args.seed, device=device, nsamples=8, dataset=args.calib_dataset)
    # print(f'{args.calib_dataset} perplexity Train {ppl_train} Test {ppl_test}')
    orig_size, orig_memory = get_model_size_and_memory(model)
    original_nozero_param_count = get_nonzero_param_count(model)
    model.eval()

    
    if os.path.isfile(os.path.abspath(os.path.join(args.save_model, 'config.json'))) and args.prune_method != "none":
        print(f"model already exists at {args.save_model}")
        # torch empty cache
        del model
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(args.save_model, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(args.save_model, use_fast=False)
        model.seqlen = model.config.max_position_embeddings 
    else:
        os.makedirs(args.save_model, exist_ok=True)
        

        if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
            device = model.hf_device_map["lm_head"]
        print("use device ", device)

        if args.sparsity_ratio != 0:
            print("pruning starts")
            if args.prune_method == "select":
                model = select_layers(args, model)
            if args.prune_method == "merge":
                model = merge_layers(args, model, tokenizer, device)
            elif args.prune_method == "wanda":
                prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "magnitude":
                prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "sparsegpt":
                prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "shortened_llm":
                model = prune_shortened_llm(args, model, tokenizer, device)
            elif args.prune_method == "bonsai":
                prune_bonsai(args, model, tokenizer, device)
                # print(model)
            elif "ablate" in args.prune_method:
                prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            elif args.prune_method == "shortgpt":
                model = prune_shortgpt(args, model, tokenizer, device)
            elif args.prune_method == "laco":
                model = prune_laco(args, model, tokenizer, device)
            elif args.prune_method == "none":
                pass
        #!
        # model.save_pretrained(args.save_model)
        # tokenizer.save_pretrained(args.save_model)

    compressed_size, compressed_memory = get_model_size_and_memory(model)
    print("*"*30)
    print(f"original model size: {orig_size / 1e9:.2f} B, compressed model size: {compressed_size / 1e9:.2f} B")
    print(f"original model memory: {orig_memory / 1024:.2f} GB, compressed model memory: {compressed_memory / 1024:.2f} GB")
    print(f"compression ratio: {compressed_size / orig_size:.2%}")
    print('model compressed \n')
    # assert 1==0
    # ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    cur_nozero_param_count = get_nonzero_param_count(model)
    sparsity_ratio_unified = 1 - cur_nozero_param_count / original_nozero_param_count
    print(f"sparsity unified {sparsity_ratio_unified:.4f}")
    print("*"*30)
    # assert 1==0
    # ################################################################
    # ppl_test = eval_ppl(args, model, tokenizer, device)
    # print(f"wikitext perplexity {ppl_test}")
    #! calculate FLOPs
    
    # torch.cuda.empty_cache()
    # model_cpu = model.to(torch.device('cpu')).float()
    # flops, macs, params = calculate_flops(
    # model=model_cpu,
    # input_shape=(1, model.seqlen),
    # transformer_tokenizer=tokenizer,
    # print_results=False
    # )
    # print(f"FLOPs: {flops}, MACs: {macs}, Params: {params}")



    if args.eval_zero_shot:
        perplexity_results = {}
        for calib_dataset in ["wikitext2", "c4"]:
            ppl_train, ppl_test = eval_ppl(model, tokenizer, seed=args.seed, device=device, nsamples=8, dataset=calib_dataset)
            perplexity_results[calib_dataset] = {'train': ppl_train, 'test': ppl_test}
            print(f'{calib_dataset} perplexity Train {ppl_train} Test {ppl_test}')
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["leaderboard_short"]#["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]#["leaderboard"]#, 
        num_shot = 0
        torch.cuda.empty_cache()
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        results_dict = results['results']
        results_dict['perplexity'] = perplexity_results
        print("********************************")
        print("zero_shot evaluation results")
        print(results_dict)

        with open(f'{args.save_model}/results.pkl', 'wb') as f:
            pickle.dump(results_dict, f)


if __name__ == '__main__':
    main()
