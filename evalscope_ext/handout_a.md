# Why This Works — Technical Handout

## Problem

Full benchmarks (315 LCB + 100 AA-LCR samples) are expensive to run for every customer evaluation. We need a pruned subset that preserves model ranking so sales can get a fast go/no-go signal.
Key insight: ranking preservation matters more than exact score match.

## LCB Approach

- **Stratified by difficulty tier**: preserves difficulty distribution — easy/medium/hard samples appear in proportion
- **Evenly-spaced within tier**: deterministic, reproducible; no randomness to manage
- **Secondary sort by question_type**: diversity within tier so no single problem type dominates
- **Metadata-only selection**: generalizes to any new model

> "The subset generalizes because selection depends on workload structure, not model behavior."

Implementation (`prune_ratio=0.1`, default):
```
n = max(1, round(len(tier) * 0.1))
step = max(1, len(tier) // n)
selected = tier[::step][:n]
```
Result: ~31 of 315 samples, all difficulty tiers represented.

## AA-LCR Approach

- **Stratified by context length**: structural discriminator for long-context tasks
  - short < 2 000 chars · medium 2 000–8 000 chars · long > 8 000 chars
- **Judge noise**: AA-LCR uses an LLM judge, so score variance includes judge randomness, not just sample difficulty. Stratifying by context length (a fixed structural property) avoids selecting samples biased by a noisy judge round.
- **Metadata-only**: same generalization guarantee as LCB — works unchanged when a new model is added

## Part B — MMMU Multimodal Probe (written design)

**Goal**: detect image encoder degradation specifically, not generic reasoning failure. These are different failure modes.

**Image types that stress the encoder**:
- Dense text in images (OCR stress)
- Circuit diagrams and technical schematics
- Scientific figures with fine axis labels
- Multi-panel images requiring spatial parsing
- Medical imaging scans
- Low-contrast or cluttered backgrounds

**Why these**: the encoder degrades on fine-detail parsing before high-level scene understanding. A model passes natural-photo QA with a weak encoder but fails on circuit diagrams. This separates encoder failure from reasoning failure.

**Selection from 12 K MMMU without loading images**:
- Filter subjects: Engineering, Medicine, Basic Science, Computer Science (naturally contain encoder-stress images)
- Prefer question types that require reading detail from the image
- Target ~200 samples (~2% of 12 K)

**Measurement**: compare probe accuracy vs. a random MMMU sample of equal size. A gap > 10 points signals encoder degradation.

## Assumptions

- Difficulty distribution in the 315/100 samples is representative of the broader task distribution
- 10% subset preserves rank order (validated on 3 shipped models)
- LLM judge is consistent across samples within a single run

## What Would Change

| Condition | Response |
|-----------|----------|
| More data available | Use IRT to score sample discrimination power empirically; reweight tiers |
| Live model endpoint | Track rank correlation over time; recalibrate if Spearman ρ drops below 0.9 |
| More time | Automate recalibration trigger on rank correlation drop; add confidence intervals |
