## Depth as a scaling vector: Simple pruning and systematic evaluation of emergent abilities in LLMs

### modifications

modified the newest version of the lm-eval-harness, after downloading the newest repo at https://github.com/huggingface/lm-evaluation-harness, replace lm-evaluation-harness/lm_eval/evaluator.py and lm-evaluation-harness/lm_eval/models/huggingface.py using the code provided in lm_eval_modify/.

Add the **leaderboard_short** folder to lm-evaluation-harness/lm_eval/tasks/.

To run PruneBench, simply add ``leaderboard_short'' to the task\_list at main.py, line 345.