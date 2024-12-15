# Adapted from https://github.com/locuslab/wanda/blob/main/lib/eval.py

# Import necessary modules
import time
import torch
import torch.nn as nn
import numpy as np

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

from collections import defaultdict
import fnmatch
def cosine_similarity_matrix(A):
    # Compute dot product for all pairs
    dot_product = np.dot(A, A.T)
    
    # Compute norms
    norms = np.linalg.norm(A, axis=1) + 1e-8
    
    # Compute similarity matrix
    similarity = dot_product / (np.outer(norms, norms))
    return similarity

# TODO Function to get layer-wise output of the model using the eval_ppl datasets.
def get_layer_output_similarity(model, tokenizer, device=torch.device("cuda:0"), dataset="c4", bsz=1):
	
    # Print status
    print(f"fetching layer outputs on {dataset}")

    # Get the test loader
    trainloader, _ = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    nsamples = len(trainloader)
    nsamples = 16  #!

    # List to store negative log likelihoods
    # nsamples = 1
    features = []
    def hook_fn(module, input, output):
        result = output[0] if isinstance(output, tuple) else output
        result = result.detach().cpu().squeeze().mean(0).numpy()
        # normalize result
        # result = result / (np.linalg+ 1e-8)
        features.append(result)

    for layer in model.model.layers:
        # print(layer)
        layer.register_forward_hook(hook_fn)
    # assert 1==0
    print(f"nsamples {nsamples}")   
    similarities = []
    with torch.no_grad():
    # Loop through each batch
        for i in range(0,nsamples,bsz):
            if i % 50 == 0:
                print(f"sample {i}")

            features = []
            # Calculate end index
            j = min(i+bsz, nsamples)

            # Prepare inputs and move to device
            # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
            this_bs = min(bsz, nsamples - i)
            inputs = torch.concat([trainloader[i + k][0].to(device) for k in range(this_bs)])

            inputs = inputs.reshape(j-i, model.seqlen)

            # Forward pass through the model
            outputs = model(inputs)
            # print(len(features), features[0].shape)
            similarity = cosine_similarity_matrix(np.stack(features, axis=0))
            similarities.append(similarity)

        torch.cuda.empty_cache()
        # print(features)


    mean_similarities = np.stack(similarities, axis=0).mean(axis=0)
    return mean_similarities
	# return train_outputs

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    dataset = "wikitext2"
    print(type(model))

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        ppl_test = eval_ppl_wikitext(model, testloader, 1, device)
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_manager = tasks.TaskManager()
    task_names = pattern_match(task_list, task_manager.all_tasks)
    print(task_names)
    
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    # model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    # if use_accelerate:
        # model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    model_args={"pretrained":model, "cache_dir":"./llm_weights", "use_accelerate":use_accelerate, 
                "add_special_tokens":add_special_tokens, "tokenizer":tokenizer}
        
    results = evaluator.simple_evaluate(
        # model="hf-causal-experimental",
        model='hf-auto',
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        # no_cache=True,
        limit=limit,
        # description_dict={},
        # decontamination_ngrams_path=None,
        check_integrity=False,
        # pretrained_model=model,
        # add_special_tokens=add_special_tokens
    )

    return results 