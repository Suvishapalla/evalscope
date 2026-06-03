import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark.meta import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench Pruned',
        tags=[Tags.CODING],
        description=(
            'Difficulty-stratified 10% subset of LiveCodeBench pre-computed results. '
            'Selection uses only structural metadata so the subset generalises to any new model.'
        ),
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        metric_list=['acc'],
        eval_split='test',
    )
)
class LiveCodeBenchPruned:
    """
    Loads, prunes, and validates pre-computed LiveCodeBench evaluation results.

    Data layout expected under data_dir:
        predictions/live_code_bench_v5__<model>.jsonl
        reviews/live_code_bench_v5__<model>.jsonl
    """

    # ------------------------------------------------------------------
    # load_samples
    # ------------------------------------------------------------------

    def load_samples(self, data_dir: str = 'Evals/Part 1') -> List[Dict[str, Any]]:
        """
        Scan data_dir/predictions/ for live_code_bench_v5__*.jsonl files,
        join each with its matching reviews file on the 'index' field, and
        return a list of per-sample dicts aggregated across all models.

        Each dict shape:
            {
                'index': <any>,
                'difficulty': <str>,       # from metadata.difficulty
                'question_type': <str>,    # from metadata.question_type
                'scores': {<model>: 0|1, ...}
            }
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f'WARNING: data_dir {data_dir!r} does not exist. Returning empty list.')
            return []

        pred_dir = data_path / 'predictions'
        review_dir = data_path / 'reviews'

        pred_files = sorted(pred_dir.glob('live_code_bench_v5__*.jsonl'))
        if not pred_files:
            print(f'WARNING: No live_code_bench_v5__*.jsonl files found in {pred_dir}')
            return []

        samples_by_index: Dict[Any, Dict] = {}

        for pred_file in pred_files:
            # Derive model name from filename
            model_name = pred_file.stem.replace('live_code_bench_v5__', '')

            review_file = review_dir / pred_file.name
            if not review_file.exists():
                print(f'WARNING: Review file not found: {review_file}. Skipping model {model_name!r}.')
                continue

            predictions = _load_jsonl_by_index(pred_file)
            reviews = _load_jsonl_by_index(review_file)

            for idx, pred in predictions.items():
                review = reviews.get(idx)
                if review is None:
                    continue

                if idx not in samples_by_index:
                    meta = pred.get('metadata', {})
                    samples_by_index[idx] = {
                        'index': idx,
                        'difficulty': meta.get('difficulty', 'unknown'),    # ASSUMED FIELD - verify against actual JSONL when data available
                        'question_type': meta.get('question_type', 'unknown'),  # ASSUMED FIELD - verify against actual JSONL when data available
                        'scores': {},
                    }

                # ASSUMED FIELD - verify against actual JSONL when data available
                samples_by_index[idx]['scores'][model_name] = int(review.get('pass', 0))

        return list(samples_by_index.values())

    # ------------------------------------------------------------------
    # prune
    # ------------------------------------------------------------------

    def prune(
        self,
        samples: List[Dict[str, Any]],
        prune_ratio: float = 0.1,
        pruning_strategy: str = 'stratified_difficulty',
    ) -> List[Dict[str, Any]]:
        """
        Select a difficulty-stratified subset.

        Selection is deterministic and depends ONLY on structural metadata
        (difficulty, question_type). Model scores are never used.
        """
        if not samples:
            print('WARNING: samples is empty. Returning empty list.')
            return []

        # Group by difficulty tier
        tiers: Dict[str, List[Dict]] = {}
        for sample in samples:
            tier = sample.get('difficulty', 'unknown')
            tiers.setdefault(tier, []).append(sample)

        selected: List[Dict] = []
        for tier_samples in tiers.values():
            # Secondary sort by question_type for diversity within tier
            tier_sorted = sorted(tier_samples, key=lambda s: s.get('question_type', ''))

            n = max(1, round(len(tier_sorted) * prune_ratio))
            step = max(1, len(tier_sorted) // n)
            tier_selected = tier_sorted[::step][:n]
            selected.extend(tier_selected)

        return selected

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------

    def validate(
        self,
        full_samples: List[Dict[str, Any]],
        pruned_samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute per-model score delta and rank-preservation between full and pruned sets.

        Returns:
            {
                'per_model': {<model>: {'full': float, 'pruned': float, 'delta': float}},
                'rank_preserved': bool,
                'pruned_count': int,
                'full_count': int,
                'warning': str | None,
            }
        """
        if not full_samples or not pruned_samples:
            return {
                'per_model': {},
                'rank_preserved': False,
                'pruned_count': len(pruned_samples),
                'full_count': len(full_samples),
                'warning': 'One or both sample lists are empty.',
            }

        all_models = _collect_models(full_samples)
        full_scores: Dict[str, float] = {}
        pruned_scores: Dict[str, float] = {}
        per_model: Dict[str, Dict] = {}

        for model in sorted(all_models):
            fs = _mean_score(full_samples, model)
            ps = _mean_score(pruned_samples, model)
            delta = ps - fs
            per_model[model] = {
                'full': round(fs, 4),
                'pruned': round(ps, 4),
                'delta': round(delta, 4),
            }
            full_scores[model] = fs
            pruned_scores[model] = ps

        full_rank = sorted(full_scores, key=full_scores.__getitem__, reverse=True)
        pruned_rank = sorted(pruned_scores, key=pruned_scores.__getitem__, reverse=True)
        rank_preserved = full_rank == pruned_rank

        warning = _build_warning(per_model, len(pruned_samples))

        return {
            'per_model': per_model,
            'rank_preserved': rank_preserved,
            'pruned_count': len(pruned_samples),
            'full_count': len(full_samples),
            'warning': warning,
        }


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _load_jsonl_by_index(path: Path) -> Dict[Any, Dict]:
    records: Dict[Any, Dict] = {}
    with open(path, 'r', encoding='utf-8') as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # File may be a Git LFS pointer or otherwise non-JSON; bail out early.
                print(f'WARNING: {path.name} does not contain valid JSON at line {lineno} '
                      f'(may be a Git LFS stub). Returning empty records.')
                return {}
            # ASSUMED FIELD - verify against actual JSONL when data available
            idx = record['index']
            records[idx] = record
    return records


def _collect_models(samples: List[Dict]) -> set:
    models: set = set()
    for s in samples:
        models.update(s.get('scores', {}).keys())
    return models


def _mean_score(samples: List[Dict], model: str) -> float:
    scores = [s['scores'][model] for s in samples if model in s.get('scores', {})]
    return sum(scores) / len(scores) if scores else 0.0


def _build_warning(per_model: Dict[str, Dict], pruned_count: int) -> Optional[str]:
    parts = []
    if per_model:
        max_delta = max(abs(v['delta']) for v in per_model.values())
        if max_delta > 0.15:
            parts.append(f'Large score delta: max |delta| = {max_delta:.4f} > 0.15')
    if pruned_count < 10:
        parts.append(f'Pruned set is very small: {pruned_count} samples')
    return '; '.join(parts) if parts else None
