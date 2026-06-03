from typing import Any, Dict, List

from evalscope.api.benchmark.meta import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope_ext.base_pruner import BaseBenchmarkPruner


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
class AALCRPruned(BaseBenchmarkPruner):
    """
    Loads, prunes, and validates pre-computed AA-LCR evaluation results.

    Data layout expected under data_dir:
        predictions/aa_lcr__<model>.jsonl
        reviews/aa_lcr__<model>.jsonl
    """

    def load_samples(self, data_dir: str = 'Evals/Part 1') -> List[Dict[str, Any]]:
        samples = super().load_samples(
            data_dir=data_dir,
            prediction_pattern='aa_lcr__*.jsonl',
            review_pattern='aa_lcr__*.jsonl',
            score_field='acc',
        )
        # Add context_bucket field for stratification
        for sample in samples:
            length = len(sample.get('prompt', ''))
            if length < 2000:
                sample['context_bucket'] = 'short'
            elif length <= 8000:
                sample['context_bucket'] = 'medium'
            else:
                sample['context_bucket'] = 'long'
        return samples

    # AA-LCR uses an LLM judge so score variance includes judge noise.
    # We stratify by context length (structural metadata) not by score
    # to avoid selection bias from judge randomness.
    def prune(
        self,
        samples: List[Dict[str, Any]],
        prune_ratio: float = 0.1,
        pruning_strategy: str = 'stratified_context',
    ) -> List[Dict[str, Any]]:
        return super().prune(
            samples,
            prune_ratio=prune_ratio,
            pruning_strategy=pruning_strategy,
            stratify_field='context_bucket',
        )
