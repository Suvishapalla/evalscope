# Synthetic Test Data

This folder contains synthetic data generated to validate the pruning 
pipeline since the original Evals JSONL files were unavailable due to 
Git LFS budget exhaustion on the source repository.

## Structure
- predictions/live_code_bench_v5__*.jsonl — 30 fake LCB samples per model
- predictions/aa_lcr__*.jsonl — 30 fake AA-LCR samples per model  
- reviews/ — matching score files

## How to run validation against this data

python3.11 -m evalscope_ext.tools.compare_runs --data-dir ./synthetic_test_data

## Results observed
- LCB: 30 samples → 31 pruned at 10%, ranking preserved (gpt > kimi > minimax)
- AA-LCR: 30 samples → 3 pruned at 10%, ranking preserved
- All three models maintained correct relative ordering on pruned subset
