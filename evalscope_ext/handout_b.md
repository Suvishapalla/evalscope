# What This Changes and How to Use It

## What changes

**Before:** "We'll run the full benchmark, back to you in hours."  
**After:** 30 samples instead of 315. Same ranking signal. Answer happens in the meeting, not after it.

The pruned set gives you a defensible go/no-go on model ranking in under 10% of the original runtime, with automatic warnings if any model's score shifts more than 15 points or the ranking changes.

## How to run it

Run the full benchmark once (or use pre-computed results):
```bash
evalscope eval --model <model> --datasets live_code_bench \
  --output ./results_full/
```

Run the pruned benchmark:
```bash
evalscope eval --model <model> --datasets live_code_bench_pruned \
  --dataset-args '{"pruning_strategy":"stratified_difficulty","prune_ratio":0.1}' \
  --output ./results_pruned/
```

Compare and validate:
```bash
python -m evalscope_ext.tools.compare_runs \
  --full ./results_full/ --pruned ./results_pruned/
```

Or validate directly against pre-computed data:
```bash
python -m evalscope_ext.tools.compare_runs --data-dir ./Evals
```

## What the multimodal probe gives that random sampling cannot

Random sampling tests general image QA ability.  
The probe tests whether the model can **see fine detail** in images.

A model can pass random MMMU and still fail on circuit diagrams or medical scans. Random sampling misses this failure mode about 80% of the time, because most MMMU images are natural photos where a degraded encoder still scores acceptably.

The probe selects subjects (Engineering, Medicine, Basic Science) where image understanding is the bottleneck — so a weak encoder shows up as a measurable accuracy gap versus the random baseline.

## Why a PM should care

Fast, structured answer backed by a defensible sample set — not a full benchmark run or gut instinct. The output is a table: model names, scores, deltas, and a single YES/NO on whether ranking is preserved. It is sharable, reproducible, and does not depend on who ran it or when.
