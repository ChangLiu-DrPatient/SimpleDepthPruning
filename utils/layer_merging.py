import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from tqdm import tqdm
import psutil
import os
import numpy as np
from huggingface_hub import login, hf_hub_download
import time
import json
import matplotlib.pyplot as plt
# import seaborn as sns
import shutil

def get_local_model_path(model_path):
    """Convert HF model path to local path"""
    base_dir = "/home/scratch/arjuncho/HF_models"
    return os.path.join(base_dir, model_path.replace('/', '_'))

def check_model_exists(local_path):
    """Check if model files exist in local path"""
    required_files = ['config.json', 'model.safetensors', 'tokenizer.json']
    return os.path.exists(local_path) and all(
        os.path.exists(os.path.join(local_path, f)) for f in required_files
    )

def load_model_with_rope_scaling_adjustment(model_path, use_auth_token=True, use_bfloat16=False):
    local_path = get_local_model_path(model_path)
    
    # Create base directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)
    
    # Check if model already exists locally
    if not check_model_exists(local_path):
        print(f"Model not found in {local_path}. Downloading...")
        try:
            # Download model files to local path
            config = AutoConfig.from_pretrained(model_path, use_auth_token=use_auth_token)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                use_auth_token=use_auth_token,
                torch_dtype=torch.bfloat16 if use_bfloat16 else None
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=use_auth_token)
            
            # Save files locally
            config.save_pretrained(local_path)
            model.save_pretrained(local_path)
            tokenizer.save_pretrained(local_path)
            print(f"Model downloaded and saved to {local_path}")
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise
    else:
        print(f"Loading model from local cache: {local_path}")
    
    try:
        with open(os.path.join(local_path, 'config.json'), 'r') as f:
            config_dict = json.load(f)
        
        if 'rope_scaling' in config_dict:
            original_rope_scaling = config_dict['rope_scaling'].copy()
            print(f"Original rope_scaling: {original_rope_scaling}")
            
            config_dict['rope_scaling'] = {
                'type': original_rope_scaling.get('rope_type', 'linear'),
                'factor': original_rope_scaling.get('factor', 1.0)
            }
            print(f"Adjusted rope_scaling: {config_dict['rope_scaling']}")
        
        config = AutoConfig.from_pretrained(None, **config_dict)
        
        if use_bfloat16:
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                config=config,
                torch_dtype=torch.bfloat16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                config=config
            )
    except Exception as e:
        print(f"Error loading model with config adjustment: {e}")
        print("Attempting to load model without config adjustment...")
        if use_bfloat16:
            model = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype=torch.bfloat16
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(local_path)
    
    return model

def plot_eigenvalue_analysis(eigenvalues, module_name, target_var=None):
    """Plot eigenvalue analysis including cumulative and percentage plots with improved scaling"""
    eigenvalues_np = eigenvalues.cpu().numpy()
    
    # Normalize eigenvalues by the maximum value
    normalized_eigenvalues = eigenvalues_np / eigenvalues_np[0]
    
    # Calculate explained variance
    total_variance = eigenvalues_np.sum()
    explained_variance_ratio = eigenvalues_np / total_variance * 100
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # # Create a figure with three subplots
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # # Plot 1: Normalized eigenvalue spectrum (log scale)
    # x_range = np.arange(1, len(normalized_eigenvalues) + 1)
    # ax1.plot(x_range, normalized_eigenvalues, 'b-', label='Normalized Eigenvalue')
    # ax1.set_title(f'Normalized Eigenvalue Spectrum\n{module_name}')
    # ax1.set_xlabel('Component Index')
    # ax1.set_ylabel('Normalized Eigenvalue')
    # ax1.set_yscale('log')
    # ax1.set_xscale('log')  # Log scale for x-axis too
    # ax1.grid(True, which="both", ls="-", alpha=0.2)
    # ax1.legend()
    
    # # Plot 2: Individual explained variance ratio
    # ax2.bar(x_range, explained_variance_ratio, alpha=0.5, color='b', 
    #         label='Individual Explained Variance')
    # ax2.set_title(f'Individual Explained Variance Ratio\n{module_name}')
    # ax2.set_xlabel('Component Index')
    # ax2.set_ylabel('Explained Variance Ratio (%)')
    # ax2.grid(True, alpha=0.2)
    # ax2.legend()
    
    # # Plot 3: Cumulative explained variance ratio
    # ax3.plot(x_range, cumulative_variance_ratio, 'r-', label='Cumulative')
    # ax3.axhline(y=95, color='g', linestyle='--', label='95% Threshold')
    # ax3.fill_between(x_range, cumulative_variance_ratio, alpha=0.3, color='r')
    # ax3.set_title(f'Cumulative Explained Variance Ratio\n{module_name}')
    # ax3.set_xlabel('Number of Components')
    # ax3.set_ylabel('Cumulative Explained Variance (%)')
    # ax3.grid(True, alpha=0.2)
    # ax3.legend()
    
    # # Adjust layout and save
    # plt.tight_layout()
    # fig.suptitle(f'Eigenvalue Analysis for {module_name}', y=1.05, fontsize=14)
    # plt.savefig(f"eigenvalue_analysis_{module_name}_{time.time()}.png", 
    #             bbox_inches='tight', dpi=300)
    #
    # plt.close()
    
    # Print detailed analysis
    if target_var:
        components_tgt = np.argmax(cumulative_variance_ratio >= target_var) + 1
        print(f"\nEigenvalue Analysis Summary for {module_name}:")
        print(f"Number of components needed for:")
        print(f"  {target_var}% variance: {components_tgt}")
        return components_tgt
    # else:
    #     components_95 = np.argmax(cumulative_variance_ratio >= 95) + 1
    #     components_80 = np.argmax(cumulative_variance_ratio >= 80) + 1
    #     components_50 = np.argmax(cumulative_variance_ratio >= 50) + 1
    #     print(f"\nEigenvalue Analysis Summary for {module_name}:")
    #     print(f"  50% variance: {components_50}")
    #     print(f"  80% variance: {components_80}")
    #     print(f"  95% variance: {components_95}")
    # print(f"Variance explained by:")
    # print(f"  First component: {explained_variance_ratio[0]:.2f}%")
    # if len(eigenvalues) > 2:
    #     print(f"  First 3 components: {cumulative_variance_ratio[2]:.2f}%")
    # if len(eigenvalues) > 5:
    #     print(f"  First 5 components: {cumulative_variance_ratio[4]:.2f}%")
    # print(f"Eigenvalue decay rate:")
    # print(f"  Ratio of 2nd to 1st: {normalized_eigenvalues[1]:.3f}")
    # if len(eigenvalues) > 2:
    #     print(f"  Ratio of 3rd to 1st: {normalized_eigenvalues[2]:.3f}")
    
def merge_params(params_list, num_components, module_name, target_var):
    """Merge parameters using SVD with enhanced eigenvalue analysis"""
    original_dtype = params_list[0].dtype
    combined = torch.stack([p.flatten() for p in params_list])
    
    # Cast to float32 for SVD
    combined_float = combined.detach().cpu().to(torch.float32)
    
    # Center the data
    mean = torch.mean(combined_float, dim=0, keepdim=True)
    centered_data = combined_float - mean
    
    # Compute SVD
    _, s, v = torch.svd(centered_data)
    
    # Plot eigenvalue analysis
    if target_var:  
        num_components = plot_eigenvalue_analysis(s, module_name, target_var=target_var)
    else: plot_eigenvalue_analysis(s, module_name, target_var=target_var)
    
    merged_layers = []
    for i in range(min(num_components, len(params_list))):
        principal_direction = v[:, i]
        projections = torch.matmul(centered_data, principal_direction)
        merged = projections.mean()
        result = (merged * principal_direction + mean.squeeze()).reshape(params_list[0].shape).to(combined.device)
        merged_layers.append(result.to(original_dtype))
    
    return merged_layers, num_components

def get_layer_module(model, model_type):
    if model_type in ["opt", "llama"]:
        return model.model.layers  #!? model.model.decoder.layers doesn't work
    elif model_type == "gemma":  
        return model.model.layers
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def merge_layer_range(model, start_layer, end_layer, model_type, num_components, target_module_name, target_var):
    layers = get_layer_module(model, model_type)

    #TODO
    all_module_names = ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj', 'self_attn.o_proj', 
                        'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj', 'input_layernorm', 'post_attention_layernorm']
    all_module_names.insert(0, all_module_names.pop(all_module_names.index(target_module_name)))   # move the target module to the front
    
    for module_name in all_module_names:
        try:

            weight_params = [getattr(layers[i], module_name.split('.')[0])
                             if '.' not in module_name else
                             getattr(getattr(layers[i], module_name.split('.')[0]), module_name.split('.')[1])
                             for i in range(start_layer, end_layer + 1)]
            if module_name == target_module_name:
                merged_weights, num_components = merge_params([p.weight.data for p in weight_params if hasattr(p, 'weight')], num_components, f"{start_layer}-{end_layer}_{module_name}", target_var)
            else:
                merged_weights, _ = merge_params([p.weight.data for p in weight_params if hasattr(p, 'weight')], num_components, f"{start_layer}-{end_layer}_{module_name}", target_var=None)
            
            if hasattr(weight_params[0], 'bias') and weight_params[0].bias is not None:
                merged_biases = merge_params([p.bias.data for p in weight_params if hasattr(p, 'bias') and p.bias is not None], num_components, f"{start_layer}-{end_layer}_{module_name}_bias", target_var=None)
            
            for i, merged_weight in enumerate(merged_weights):
                if '.' in module_name:
                    getattr(getattr(layers[start_layer + i], module_name.split('.')[0]), module_name.split('.')[1]).weight.data = merged_weight
                    if hasattr(weight_params[0], 'bias') and weight_params[0].bias is not None:
                        getattr(getattr(layers[start_layer + i], module_name.split('.')[0]), module_name.split('.')[1]).bias.data = merged_biases[i]
                else:
                    getattr(layers[start_layer + i], module_name).weight.data = merged_weight
                    if hasattr(weight_params[0], 'bias') and weight_params[0].bias is not None:
                        getattr(layers[start_layer + i], module_name).bias.data = merged_biases[i]
        except AttributeError:
            print(f"Skipping {module_name} as it's not present in this model architecture.")

    return model, num_components

def merge_multiple_ranges(model, ranges, model_type, num_components, target_module_name, target_var):
    num_components_list = []
    for start, end in ranges:
        model, num_components = merge_layer_range(model, start, end, model_type, num_components, target_module_name, target_var)
        num_components_list.append(num_components)
    return model, num_components_list
#!
def update_model_after_merge(model, model_type, merge_ranges, num_components):  #TODO
    """Update model configuration and layers after merging"""
    layers = get_layer_module(model, model_type)
    
    # Calculate which layers to keep
    if type(num_components) == int:
        num_components = [num_components] * len(merge_ranges)
    # layers_to_keep = set(range(len(layers))) - set(
    #     i for start, end in merge_ranges for i in range(start + num_components, end + 1)
    # )
    layers_to_keep = set(range(len(layers))) - set(
        i for j in range(len(merge_ranges)) for i in range(merge_ranges[j][0]+num_components[j], merge_ranges[j][1]+1))
    new_layers = [layer for i, layer in enumerate(layers) if i in layers_to_keep]
    
    # Update the layers
    if model_type in ["opt", "llama"]:
        # model.model.decoder.layers = torch.nn.ModuleList(new_layers)
        model.model.layers = torch.nn.ModuleList(new_layers) #!?
    elif model_type == "gemma":
        model.model.layers = torch.nn.ModuleList(new_layers)
    
    # Update config
    model.config.num_hidden_layers = len(new_layers)
    
    # Update layer indices for attention layers
    for idx, layer in enumerate(new_layers):
        if hasattr(layer, 'self_attn'):
            layer.self_attn.layer_idx = idx
    
    # Force regeneration of cache on next forward pass
    if hasattr(model, '_hf_hook'):
        model._hf_hook = None
    
    return model
#!
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def measure_inference_time(model, tokenizer, input_text, device, num_runs=10):
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Warm-up run with error handling
        try:
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=20,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
        except Exception as e:
            print(f"Warning: Warm-up generation failed: {str(e)}")
            print("Attempting generation without KV cache...")
            model.generate(
                **inputs,
                max_new_tokens=20,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
            )
        
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=20,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        return avg_time

def evaluate_multiple_choice(model, tokenizer, dataset, device):
    correct = 0
    total = 0
    total_time = 0

    model.eval()
    for item in tqdm(dataset):
        question = item['question'] if 'question' in item else item.get('goal', '')
        choices = item['choices']['text'] if 'choices' in item else [item.get('sol1', ''), item.get('sol2', '')]
        label = item.get('answerKey', item.get('label', 0))

        inputs = tokenizer([f"{question}\nAnswer: {choice}" for choice in choices], return_tensors="pt", padding=True).to(device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()
        total_time += end_time - start_time

        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        predicted = torch.argmax(probs.mean(dim=1)).item()

        if isinstance(label, str):
            correct += (ord(choices[predicted][0].lower()) - ord('a')) == (ord(label.lower()) - ord('a'))
        else:
            correct += predicted == label
        total += 1

    accuracy = correct / total
    avg_inference_time = total_time / total
    return accuracy, avg_inference_time

def evaluate_mmlu(model, tokenizer, device, subjects=None):
    if subjects is None:
        subjects = ['anatomy', 'world_religions', 'philosophy']

    total_correct = 0
    total_questions = 0
    total_time = 0

    for subject in subjects:
        dataset = load_dataset("cais/mmlu", subject, split="test")
        correct = 0

        for item in tqdm(dataset, desc=f"Evaluating {subject}"):
            question = item['question']
            choices = item['choices']
            answer = item['answer']

            inputs = tokenizer([f"{question}\nAnswer: {choice}" for choice in choices], return_tensors="pt", padding=True).to(device)

            start_time = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
            end_time = time.time()
            total_time += end_time - start_time

            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            predicted = torch.argmax(probs.mean(dim=1)).item()

            if predicted == answer:
                correct += 1

        total_correct += correct
        total_questions += len(dataset)
        print(f"{subject} accuracy: {correct / len(dataset):.4f}")

    overall_accuracy = total_correct / total_questions
    avg_inference_time = total_time / total_questions
    return overall_accuracy, avg_inference_time

def main(args):
    # Login to Hugging Face Hub
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        print("Warning: HF_TOKEN environment variable not set. Some operations may fail.")
    
    # Set up device based on user input
    if torch.cuda.is_available():
        if args.gpu_id >= torch.cuda.device_count():
            raise ValueError(f"GPU {args.gpu_id} not available. Available GPUs: 0 to {torch.cuda.device_count()-1}")
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    print(f"Loading {args.model_type} model...")
    model = load_model_with_rope_scaling_adjustment(args.model_path, use_bfloat16=args.use_bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(get_local_model_path(args.model_path))

    if args.use_bfloat16:
        print("Model loaded in bfloat16 precision")

    original_size = get_model_size(model)
    print(f"Original model size: {original_size:.2f} MB")

    merge_ranges = [tuple(map(int, r.split('-'))) for r in args.merge_ranges]
    print(f"Merging layer ranges: {merge_ranges}")
    model, num_components_list = merge_multiple_ranges(model, merge_ranges, args.model_type, args.num_components, args.target_module_name, args.target_var)
    print(f'num_components_list: {num_components_list}')
    print("Updating model configuration after merging...")
    model = update_model_after_merge(model, args.model_type, merge_ranges, num_components_list)

    merged_size = get_model_size(model)
    print(f"Merged model size: {merged_size:.2f} MB")
    print(f"Size reduction: {(original_size - merged_size) / original_size * 100:.2f}%")

    # Save merged model to HF_models directory
    merged_model_name = f"{args.model_type}-merged-custom"
    merged_model_path = get_local_model_path(merged_model_name)
    os.makedirs(merged_model_path, exist_ok=True)
    
    print(f"Saving the modified model to {merged_model_path}...")
    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)

    print(f"Moving model to device: {device}")
    model.to(device)

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Measure inference time
    input_text = "In a world of rapid technological advancements, what role does ethics play in guiding innovation?"
    merged_inference_time = measure_inference_time(model, tokenizer, input_text, device)
    print(f"Merged model average inference time: {merged_inference_time:.4f} seconds")

    if args.benchmark in ["piqa", "commonsense_qa"]:
        dataset = load_dataset(args.benchmark, split="validation")
        print(f"Evaluating merged model on {args.benchmark}...")
        merged_accuracy, merged_avg_time = evaluate_multiple_choice(model, tokenizer, dataset, device)
    elif args.benchmark == "mmlu":
        print("Evaluating merged model on MMLU...")
        merged_accuracy, merged_avg_time = evaluate_mmlu(model, tokenizer, device)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")

    print(f"Merged model accuracy on {args.benchmark}: {merged_accuracy:.4f}")
    print(f"Merged model average inference time on {args.benchmark}: {merged_avg_time:.4f} seconds")

    print(f"Loading original {args.model_type} model for comparison...")
    original_model = load_model_with_rope_scaling_adjustment(args.model_path, use_bfloat16=args.use_bfloat16).to(device)

    original_inference_time = measure_inference_time(original_model, tokenizer, input_text, device)
    print(f"Original model average inference time: {original_inference_time:.4f} seconds")

    print(f"Evaluating original model on {args.benchmark}...")
    if args.benchmark in ["piqa", "commonsense_qa"]:
        original_accuracy, original_avg_time = evaluate_multiple_choice(original_model, tokenizer, dataset, device)
    elif args.benchmark == "mmlu":
        original_accuracy, original_avg_time = evaluate_mmlu(original_model, tokenizer, device)

    print(f"Original model accuracy on {args.benchmark}: {original_accuracy:.4f}")
    print(f"Original model average inference time on {args.benchmark}: {original_avg_time:.4f} seconds")

    print(f"Accuracy difference: {merged_accuracy - original_accuracy:.4f}")
    print(f"Inference time difference: {original_avg_time - merged_avg_time:.4f} seconds")
    print(f"Inference time reduction: {(original_avg_time - merged_avg_time) / original_avg_time * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge model layers and evaluate on a benchmark.")
    parser.add_argument("--model_type", choices=["opt", "llama", "gemma"], required=True, help="Type of model to use")
    parser.add_argument("--model_path", required=True, help="Path or identifier of the model to use")
    parser.add_argument("--merge_ranges", nargs="+", required=True, help="Ranges of layers to merge, e.g., '2-12 14-17 18-19'")
    parser.add_argument("--benchmark", choices=["piqa", "commonsense_qa", "mmlu"], required=True, help="Benchmark to evaluate on")
    parser.add_argument("--use_bfloat16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--num_components", type=int, default=1, help="Number of principal components to use for merging")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use (default: 0)")
    args = parser.parse_args()

    main(args)