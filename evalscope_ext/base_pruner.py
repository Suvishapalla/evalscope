import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class BaseBenchmarkPruner:
    """Generic base class for loading, pruning, and validating benchmark eval results."""

    def load_samples(
        self,
        data_dir: str,
        prediction_pattern: str,
        review_pattern: str,
        score_field: str,
        join_field: str = 'index',
    ) -> List[Dict[str, Any]]:
        """
        Scan data_dir/predictions/ for files matching prediction_pattern glob,
        join with matching reviews from data_dir/reviews/, and return merged samples.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f'WARNING: data_dir {data_dir!r} does not exist. Returning empty list.')
            return []

        pred_dir = data_path / 'predictions'
        review_dir = data_path / 'reviews'

        pred_files = sorted(pred_dir.glob(prediction_pattern))
        if not pred_files:
            print(f'WARNING: No {prediction_pattern!r} files found in {pred_dir}')
            return []

        samples_by_index: Dict[Any, Dict] = {}

        for pred_file in pred_files:
            review_file = review_dir / pred_file.name
            if not review_file.exists():
                print(f'WARNING: Review file not found: {review_file}. Skipping.')
                continue

            predictions = _load_jsonl_by_field(pred_file, join_field)
            reviews = _load_jsonl_by_field(review_file, join_field)

            if not predictions or not reviews:
                continue

            for idx, pred in predictions.items():
                review = reviews.get(idx)
                if review is None:
                    continue

                if idx not in samples_by_index:
                    sample = dict(pred)
                    sample[join_field] = idx
                    sample.setdefault('scores', {})
                    samples_by_index[idx] = sample

                model_name = _model_name_from_file(pred_file, prediction_pattern)
                samples_by_index[idx]['scores'][model_name] = int(review.get(score_field, 0))

        return list(samples_by_index.values())

    def prune(
        self,
        samples: List[Dict[str, Any]],
        prune_ratio: float = 0.1,
        pruning_strategy: str = 'stratified',
        stratify_field: Optional[str] = None,
        secondary_sort_field: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Select a stratified subset using evenly-spaced indexing within each group.

        If stratify_field is None, treats all samples as one group.
        """
        if not samples:
            print('WARNING: samples is empty. Returning empty list.')
            return []

        if stratify_field is None:
            groups: Dict[str, List[Dict]] = {'all': list(samples)}
        else:
            groups = {}
            for sample in samples:
                key = str(sample.get(stratify_field, 'unknown'))
                groups.setdefault(key, []).append(sample)

        selected: List[Dict] = []
        for group_samples in groups.values():
            if secondary_sort_field is not None:
                group_samples = sorted(group_samples, key=lambda s: s.get(secondary_sort_field, ''))
            n = max(1, round(len(group_samples) * prune_ratio))
            step = max(1, len(group_samples) // n)
            selected.extend(group_samples[::step][:n])

        return selected

    def validate(
        self,
        full_samples: List[Dict[str, Any]],
        pruned_samples: List[Dict[str, Any]],
        score_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute per-model score delta and rank-preservation between full and pruned sets.
        score_fields: field names to average as scores; defaults to 'scores' dict keys.
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
# Shared helpers
# ------------------------------------------------------------------

def _load_jsonl_by_field(path: Path, field: str) -> Dict[Any, Dict]:
    records: Dict[Any, Dict] = {}
    with open(path, 'r', encoding='utf-8') as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f'WARNING: {path.name} does not contain valid JSON at line {lineno} '
                      f'(may be a Git LFS stub). Returning empty records.')
                return {}
            idx = record.get(field)
            if idx is not None:
                records[idx] = record
    return records


def _model_name_from_file(pred_file: Path, pattern: str) -> str:
    """Extract model name by stripping the literal prefix/suffix from the glob pattern."""
    stem = pred_file.stem
    prefix = pattern.split('*')[0].rstrip('_')
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix):].lstrip('_')
    return stem or pred_file.stem


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
