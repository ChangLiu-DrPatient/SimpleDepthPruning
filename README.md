## Documentation for PCA-based LLM Pruning

### modifications

11/24/2024: 
modified the newest version of the lm-eval-harness, after downloading the newest repo at https://github.com/huggingface/lm-evaluation-harness, replace lm_eval/evaluator.py and lm_eval/models/huggingface.py using the code provided in lm_eval_modify/. 

To run the leaderboard, simply add ``leaderboard'' to the task\_list at main.py, line 99.