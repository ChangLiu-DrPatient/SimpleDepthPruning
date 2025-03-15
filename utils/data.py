# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    # print('loading dataset...')
    traindata = load_dataset('allenai/c4',  data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
    print('loaded c4 dataset', seqlen)

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            # print(trainenc.input_ids.shape[1], seqlen)
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :seqlen]  #! (256 * seqlen?)
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Load and process bookcorpus dataset
def get_bookcorpus(nsamples, seed, seqlen, tokenizer):
    # Load the BookCorpus dataset (assuming the 'bookcorpus' identifier works)
    traindata = load_dataset("bookcorpus", split="train[:10%]")
    valdata  = load_dataset("bookcorpus", split="train[10%:15%]")
    print("loaded bookcorpus dataset", seqlen)

    # Generate samples from the training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            idx = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[idx]['text'], return_tensors='pt')
            # print(trainenc.input_ids.shape[1])
            # Ensure the encoded text is long enough
            if trainenc.input_ids.shape[1] > seqlen:
                break
        start_idx = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        end_idx = start_idx + seqlen
        inp = trainenc.input_ids[:, start_idx:end_idx]
        tar = inp.clone()
        # Mask out all targets except the final token prediction
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare a validation sample by concatenating a subset of texts.
    # Here, we use the first 1100 examples from traindata.
    valid_texts = valdata[:1100]['text']
    val_text = " ".join(valid_texts)
    valenc = tokenizer(val_text, return_tensors='pt')
    valenc = valenc.input_ids[:, :seqlen]
    # print(valenc)
    valenc = TokenizerWrapper(valenc)  # Ensure this wrapper is defined in your code.
    
    return trainloader, valenc

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if 'bookcorpus' in name:
        return get_bookcorpus(nsamples, seed, seqlen, tokenizer)