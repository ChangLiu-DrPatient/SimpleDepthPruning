## Depth as a scaling vector: Simple pruning and systematic evaluation of emergent abilities in LLMs

### modifications

modified the newest version of the lm-eval-harness, after downloading the newest repo at https://github.com/huggingface/lm-evaluation-harness, replace lm-evaluation-harness/lm_eval/evaluator.py and lm-evaluation-harness/lm_eval/models/huggingface.py using the code provided in lm_eval_modify/.

Add the **leaderboard_short** folder to lm-evaluation-harness/lm_eval/tasks/.

## deployment

The pruning & evaluation scripts are in scripts.txt and scripts_llama2.txt