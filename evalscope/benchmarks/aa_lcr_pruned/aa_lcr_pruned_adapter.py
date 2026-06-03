import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark.meta import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags


@register_benchmark(
    BenchmarkMeta(
        name='aa_lcr_pruned',
        pretty_name='AA-LCR Pruned',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description=(
            'Context-length-stratified 10% subset of AA-LCR pre-computed results. '
            'Selection uses only structural metadata so the subset generalises to any new model.'
        ),
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        eval_split='test',
    )
)
class AALCRPruned:
    """
    Loads, prunes, and validates pre-computed AA-LCR evaluation results.

    Data layout expected under data_dir:
        predictions/aa_lcr__<model>.jsonl
        reviews/aa_lcr__<model>.jsonl
    """

    # ------------------------------------------------------------------
    # load_samples
    # ------------------------------------------------------------------

    def load_samples(self, data_dir: str = 'Evals/Part 1') -> List[Dict[str, Any]]:
        """
        Scan data_dir/predictions/ for aa_lcr__*.jsonl files,
        join each with its matching reviews file on the 'index' field, and
        return a list of per-sample dicts aggregated across all models.

        Each dict shape:
            {
                'index': <any>,
                'prompt': <str>,           # full prompt text, used for context-length bucketing
                'scores': {<model>: 0|1, ...}
            }
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f'WARNING: data_dir {data_dir!r} does not exist. Returning empty list.')
            return []

        pred_dir = data_path / 'predictions'
        review_dir = data_path / 'reviews'

        pred_files = sorted(pred_dir.glob('aa_lcr__*.jsonl'))
        if not pred_files:
            print(f'WARNING: No aa_lcr__*.jsonl files found in {pred_dir}')
            return []

        samples_by_index: Dict[Any, Dict] = {}

        for pred_file in pred_files:
            # Derive model name from filename
            model_name = pred_file.stem.replace('aa_lcr__', '')

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
                    # ASSUMED FIELD - verify against actual JSONL when data available
                    prompt_text = pred.get('prompt', '')
                    samples_by_index[idx] = {
                        'index': idx,
                        'prompt': prompt_text,
                        'scores': {},
                    }

                # ASSUMED FIELD - verify against actual JSONL when data available
                samples_by_index[idx]['scores'][model_name] = int(review.get('acc', 0))

        return list(samples_by_index.values())

    # ------------------------------------------------------------------
    # prune
    # ------------------------------------------------------------------

    # AA-LCR uses an LLM judge so score variance includes judge noise.
    # We stratify by context length (structural metadata) not by score
    # to avoid selection bias from judge randomness.
    def prune(
        self,
        samples: List[Dict[str, Any]],
        prune_ratio: float = 0.1,
        pruning_strategy: str = 'stratified_context',
    ) -> List[Dict[str, Any]]:
        """
        Select a context-length-stratified subset.

        Buckets:
            short:  < 2000 chars
            medium: 2000-8000 chars
            long:   > 8000 chars

        Selection is deterministic and depends ONLY on structural metadata
        (prompt length). Model scores are never used.
        """
        if not samples:
            print('WARNING: samples is empty. Returning empty list.')
            return []

        # Bucket by context length of the 'prompt' field
        buckets: Dict[str, List[Dict]] = {'short': [], 'medium': [], 'long': []}
        for sample in samples:
            length = len(sample.get('prompt', ''))  # ASSUMED FIELD - verify against actual JSONL when data available
            if length < 2000:
                buckets['short'].append(sample)
            elif length <= 8000:
                buckets['medium'].append(sample)
            else:
                buckets['long'].append(sample)

        selected: List[Dict] = []
        for bucket_samples in buckets.values():
            if not bucket_samples:
                continue
            n = max(1, round(len(bucket_samples) * prune_ratio))
            step = max(1, len(bucket_samples) // n)
            selected.extend(bucket_samples[::step][:n])

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
