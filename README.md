## Documentation for PCA-based LLM Pruning

### modifications

11/24/2024:

modified the newest version of the lm-eval-harness, after downloading the newest repo at https://github.com/huggingface/lm-evaluation-harness, replace lm_eval/evaluator.py and lm_eval/models/huggingface.py using the code provided in lm_eval_modify/.

To run the leaderboard, simply add ``leaderboard'' to the task\_list at main.py, line 99.

11/25/2024:

integrated Bonsai as a baseline. The "--dataset" flag in the Bonsai repo is modified to "--calib_dataset" and the "--prune_method" flag is modified to "--bonsai_prune_method". A sample run would be:

```
CUDA_VISIBLE_DEVICES=1 python main.py --model meta-llama/Llama-2-7b-hf --calib_dataset wikitext2 --sparsity_ratio 0.5 --masks_per_iter 6 --nsamples 1 --save outdir --prune_frac 0.1 --bsz 1 --prune_method bonsai --eval_zero_shot
```