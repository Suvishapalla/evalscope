## Cerebras Benchmark Pruning Extension

Developed against commit: d1cc7ef9bdcd5d4ce5f77dc91a17ff484a1da9e6

### Install
    pip install -e .

### Run full benchmark
    evalscope eval --model <model> --datasets live_code_bench --output ./results_full/

### Run pruned benchmark
    evalscope eval --model <model> --datasets live_code_bench_pruned --dataset-args '{"pruning_strategy":"stratified_difficulty","prune_ratio":0.1}' --output ./results_pruned/

### Compare results
    python -m evalscope_ext.tools.compare_runs --full ./results_full/ --pruned ./results_pruned/

### Validate on pre-computed data
    python -m evalscope_ext.tools.compare_runs --data-dir ./Evals

