from typing import Any, Dict, List

from evalscope.api.benchmark.meta import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope_ext.base_pruner import BaseBenchmarkPruner


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
class LiveCodeBenchPruned(BaseBenchmarkPruner):
    """
    Loads, prunes, and validates pre-computed LiveCodeBench evaluation results.

    Data layout expected under data_dir:
        predictions/live_code_bench_v5__<model>.jsonl
        reviews/live_code_bench_v5__<model>.jsonl
    """

    def load_samples(self, data_dir: str = 'Evals/Part 1') -> List[Dict[str, Any]]:
        samples = super().load_samples(
            data_dir=data_dir,
            prediction_pattern='live_code_bench_v5__*.jsonl',
            review_pattern='live_code_bench_v5__*.jsonl',
            score_field='pass',
        )
        # Promote metadata fields to top-level for stratification
        for sample in samples:
            meta = sample.get('metadata', {})
            sample.setdefault('difficulty', meta.get('difficulty', 'unknown'))
            sample.setdefault('question_type', meta.get('question_type', 'unknown'))
        return samples

    def prune(
        self,
        samples: List[Dict[str, Any]],
        prune_ratio: float = 0.1,
        pruning_strategy: str = 'stratified_difficulty',
    ) -> List[Dict[str, Any]]:
        return super().prune(
            samples,
            prune_ratio=prune_ratio,
            pruning_strategy=pruning_strategy,
            stratify_field='difficulty',
            secondary_sort_field='question_type',
        )
