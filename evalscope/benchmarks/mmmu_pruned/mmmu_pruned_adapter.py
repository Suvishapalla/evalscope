import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evalscope.api.benchmark.meta import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope_ext.base_pruner import BaseBenchmarkPruner


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_pruned',
        pretty_name='MMMU Pruned',
        tags=[Tags.KNOWLEDGE, Tags.REASONING],
        description=(
            'Encoder-stress-stratified 10% subset of MMMU pre-computed results. '
            'Prioritises visually dense subjects to surface multimodal encoder degradation.'
        ),
        dataset_id='opencompass/MMMU',
        metric_list=['acc'],
        eval_split='validation',
    )
)
class MMMUPruned(BaseBenchmarkPruner):
    """
    Loads, prunes, and validates pre-computed MMMU evaluation results.

    Data layout expected under data_dir:
        predictions/glm-4.5v-fp8/mmmu_<Subject>.jsonl
        reviews/glm-4.5v-fp8/mmmu_<Subject>.jsonl
    """

    ENCODER_STRESS_SUBJECTS = {
        'Architecture_and_Engineering',
        'Basic_Medical_Science',
        'Clinical_Medicine',
        'Diagnostics_and_Laboratory_Medicine',
        'Electronics',
        'Energy_and_Power',
        'Materials',
        'Computer_Science',
        'Chemistry',
        'Biology',
    }

    def load_samples(self, data_dir: str = 'Evals/MMMU') -> List[Dict[str, Any]]:
        """
        Load all per-subject JSONL files from predictions/<model>/,
        join with matching reviews on the index field, annotate subject
        and is_encoder_stress fields.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f'WARNING: data_dir {data_dir!r} does not exist. Returning empty list.')
            return []

        pred_root = data_path / 'predictions'
        review_root = data_path / 'reviews'

        pred_dirs = sorted([d for d in pred_root.iterdir() if d.is_dir()]) if pred_root.exists() else []
        if not pred_dirs:
            print(f'WARNING: No model directories found under {pred_root}')
            return []

        all_samples: List[Dict[str, Any]] = []

        for model_dir in pred_dirs:
            model_name = model_dir.name
            review_dir = review_root / model_name

            pred_files = sorted(model_dir.glob('mmmu_*.jsonl'))
            if not pred_files:
                print(f'WARNING: No mmmu_*.jsonl files found in {model_dir}')
                continue

            for pred_file in pred_files:
                subject = pred_file.stem.replace('mmmu_', '')
                review_file = review_dir / pred_file.name

                predictions = _load_jsonl_by_index(pred_file)
                if not predictions:
                    continue

                reviews: Dict[Any, Dict] = {}
                if review_file.exists():
                    reviews = _load_jsonl_by_index(review_file)

                for idx, pred in predictions.items():
                    review = reviews.get(idx, {})
                    sample: Dict[str, Any] = dict(pred)
                    sample['index'] = idx
                    sample['subject'] = subject
                    sample['is_encoder_stress'] = subject in self.ENCODER_STRESS_SUBJECTS
                    sample['scores'] = {model_name: int(review.get('acc', 0))} if review else {}
                    all_samples.append(sample)

        return all_samples

    def prune(
        self,
        samples: List[Dict[str, Any]],
        prune_ratio: float = 0.1,
        pruning_strategy: str = 'encoder_stress_stratified',
    ) -> List[Dict[str, Any]]:
        """
        Two-phase selection:
          Phase 1: 70% of budget from encoder_stress subjects (evenly spaced)
          Phase 2: 30% of budget from non-stress subjects (evenly spaced)
        """
        if not samples:
            print('WARNING: samples is empty. Returning empty list.')
            return []

        total_budget = max(1, round(len(samples) * prune_ratio))
        stress_budget = max(1, round(total_budget * 0.7))
        non_stress_budget = max(1, total_budget - stress_budget)

        stress = [s for s in samples if s.get('is_encoder_stress')]
        non_stress = [s for s in samples if not s.get('is_encoder_stress')]

        def _evenly_spaced(pool: List[Dict], n: int) -> List[Dict]:
            if not pool:
                return []
            n = min(n, len(pool))
            step = max(1, len(pool) // n)
            return pool[::step][:n]

        return _evenly_spaced(stress, stress_budget) + _evenly_spaced(non_stress, non_stress_budget)

    def validate(
        self,
        full_samples: List[Dict[str, Any]],
        pruned_samples: List[Dict[str, Any]],
        score_fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute per-subject accuracy and encoder_stress vs non_stress accuracy gap.
        A large negative stress_gap signals encoder degradation.
        """
        def _accuracy(pool: List[Dict]) -> float:
            if not pool:
                return 0.0
            hits = sum(1 for s in pool if any(v == 1 for v in s.get('scores', {}).values()))
            return hits / len(pool)

        def _by_subject(pool: List[Dict]) -> Dict[str, float]:
            by_subj: Dict[str, List] = {}
            for s in pool:
                by_subj.setdefault(s.get('subject', 'unknown'), []).append(s)
            return {subj: round(_accuracy(items), 4) for subj, items in sorted(by_subj.items())}

        pruned_stress = [s for s in pruned_samples if s.get('is_encoder_stress')]
        pruned_non_stress = [s for s in pruned_samples if not s.get('is_encoder_stress')]

        stress_acc = _accuracy(pruned_stress)
        non_stress_acc = _accuracy(pruned_non_stress)
        stress_gap = round(stress_acc - non_stress_acc, 4)

        return {
            'per_subject': {
                'full': _by_subject(full_samples),
                'pruned': _by_subject(pruned_samples),
            },
            'stress_accuracy': round(stress_acc, 4),
            'non_stress_accuracy': round(non_stress_acc, 4),
            'stress_gap': stress_gap,
            'pruned_count': len(pruned_samples),
            'full_count': len(full_samples),
            'warning': 'Encoder degradation detected (stress_gap < -0.1)' if stress_gap < -0.1 else None,
        }


def _load_jsonl_by_index(path: Path) -> Dict[Any, Dict]:
    """Load a JSONL file keyed by the 'index' field. Returns {} on LFS stubs."""
    records: Dict[Any, Dict] = {}
    with open(path, 'r', encoding='utf-8') as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f'WARNING: {path.name} is not valid JSON at line {lineno} '
                      f'(may be a Git LFS stub). Skipping file.')
                return {}
            idx = record.get('index')
            if idx is not None:
                records[idx] = record
    return records
