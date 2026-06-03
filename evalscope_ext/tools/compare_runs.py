"""
compare_runs.py — validate LCB and AA-LCR pruned subsets against full runs.

Usage:
    # Validate against pre-computed data directory:
    python -m evalscope_ext.tools.compare_runs --data-dir ./Evals

    # Compare two pre-computed result directories:
    python -m evalscope_ext.tools.compare_runs --full ./results_full --pruned ./results_pruned
"""
import argparse
import sys
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------

def _print_table(title: str, result: Dict[str, Any]) -> None:
    print(f'\n=== {title} ===')

    per_model = result.get('per_model', {})
    if not per_model:
        print('  (no model data)')
    else:
        col_w = max(len(m) for m in per_model) + 2
        header = f"{'Model':<{col_w}}{'Full Score':>12}{'Pruned Score':>14}{'Delta':>10}{'Rank OK':>10}"
        print(header)
        print('-' * len(header))

        full_rank = sorted(per_model, key=lambda m: per_model[m]['full'], reverse=True)
        pruned_rank = sorted(per_model, key=lambda m: per_model[m]['pruned'], reverse=True)

        for model in sorted(per_model):
            d = per_model[model]
            rank_ok = 'YES' if full_rank.index(model) == pruned_rank.index(model) else 'NO'
            print(
                f"{model:<{col_w}}"
                f"{d['full']:>12.2f}"
                f"{d['pruned']:>14.2f}"
                f"{d['delta']:>+10.2f}"
                f"{'  ' + rank_ok:>10}"
            )

    rank_label = 'YES' if result.get('rank_preserved') else 'NO'
    print(f"Ranking preserved: {rank_label}")

    pruned = result.get('pruned_count', 0)
    full = result.get('full_count', 0)
    # Guard against non-integer counts (e.g. '?' placeholder from run_dirs mode)
    try:
        pct = f'{int(pruned) / int(full) * 100:.0f}%' if full and int(full) > 0 else 'N/A'
    except (TypeError, ValueError):
        pct = 'N/A'
    print(f'Pruned set size: {pruned} / {full} ({pct})')

    warning = result.get('warning')
    if warning:
        print(f'WARNING: {warning}')


def _print_mmmu_table(title: str, result: Dict[str, Any]) -> None:
    print(f'\n=== {title} ===')
    print(f"Encoder-stress accuracy : {result.get('stress_accuracy', 0):.4f}")
    print(f"Non-stress accuracy     : {result.get('non_stress_accuracy', 0):.4f}")
    print(f"Stress gap              : {result.get('stress_gap', 0):+.4f}  "
          f"({'degradation signal' if result.get('stress_gap', 0) < -0.1 else 'ok'})")

    per_subject = result.get('per_subject', {}).get('pruned', {})
    if per_subject:
        print('\nPer-subject accuracy (pruned):')
        col_w = max(len(s) for s in per_subject) + 2
        for subj, acc in sorted(per_subject.items()):
            print(f"  {subj:<{col_w}} {acc:.4f}")

    pruned = result.get('pruned_count', 0)
    full = result.get('full_count', 0)
    try:
        pct = f'{int(pruned) / int(full) * 100:.0f}%' if full and int(full) > 0 else 'N/A'
    except (TypeError, ValueError):
        pct = 'N/A'
    print(f'Pruned set size: {pruned} / {full} ({pct})')

    warning = result.get('warning')
    if warning:
        print(f'WARNING: {warning}')


def _print_warnings(result: Dict[str, Any], label: str) -> None:
    per_model = result.get('per_model', {})
    if per_model:
        max_delta = max(abs(v['delta']) for v in per_model.values())
        if max_delta > 0.15:
            print(f'WARNING [{label}]: score delta exceeds 0.15 (max={max_delta:.4f})')
    pruned_count = result.get('pruned_count', 99)
    try:
        if int(pruned_count) < 10:
            print(f'WARNING [{label}]: pruned set has fewer than 10 samples ({pruned_count})')
    except (TypeError, ValueError):
        pass


# ------------------------------------------------------------------
# Data-dir mode: instantiate pruners, load, validate
# ------------------------------------------------------------------

def run_data_dir(data_dir: str) -> None:
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from evalscope.benchmarks.live_code_bench_pruned.live_code_bench_pruned_adapter import LiveCodeBenchPruned
    from evalscope.benchmarks.aa_lcr_pruned.aa_lcr_pruned_adapter import AALCRPruned
    from evalscope.benchmarks.mmmu_pruned.mmmu_pruned_adapter import MMMUPruned

    lcb = LiveCodeBenchPruned()
    lcb_full = lcb.load_samples(data_dir)
    lcb_pruned = lcb.prune(lcb_full, prune_ratio=0.1)
    lcb_result = lcb.validate(lcb_full, lcb_pruned)
    _print_table('LiveCodeBench Pruned Validation', lcb_result)
    _print_warnings(lcb_result, 'LCB')

    aalcr = AALCRPruned()
    aalcr_full = aalcr.load_samples(data_dir)
    aalcr_pruned = aalcr.prune(aalcr_full, prune_ratio=0.1)
    aalcr_result = aalcr.validate(aalcr_full, aalcr_pruned)
    _print_table('AA-LCR Pruned Validation', aalcr_result)
    _print_warnings(aalcr_result, 'AA-LCR')

    from pathlib import Path as _Path
    _evals_root = _Path(data_dir)
    _mmmu_dir = _evals_root / 'MMMU' if (_evals_root / 'MMMU').exists() else _evals_root
    mmmu = MMMUPruned()
    mmmu_full = mmmu.load_samples(str(_mmmu_dir))
    mmmu_pruned = mmmu.prune(mmmu_full, prune_ratio=0.1)
    mmmu_result = mmmu.validate(mmmu_full, mmmu_pruned)
    _print_mmmu_table('MMMU Pruned Validation', mmmu_result)
    if mmmu_result.get('warning'):
        print(f"WARNING [MMMU]: {mmmu_result['warning']}")


# ------------------------------------------------------------------
# Full/pruned directory mode (future: load pre-computed result JSONs)
# ------------------------------------------------------------------

def run_dirs(full_dir: str, pruned_dir: str) -> None:
    import json
    from pathlib import Path

    def _load_results(path: str) -> Optional[Dict]:
        p = Path(path)
        candidates = list(p.glob('*.json')) + list(p.glob('results*.json'))
        if not candidates:
            print(f'WARNING: No result JSON files found in {path}')
            return None
        with open(candidates[0]) as fh:
            return json.load(fh)

    full_data = _load_results(full_dir)
    pruned_data = _load_results(pruned_dir)

    if full_data is None or pruned_data is None:
        print('Cannot compare: one or both result directories are missing result files.')
        return

    # Build a simple per_model comparison from loaded JSONs
    # Skip metadata keys (non-dict values like __count__)
    per_model: Dict[str, Dict] = {}
    models = {k for k in set(full_data.keys()) & set(pruned_data.keys())
              if isinstance(full_data[k], dict) and isinstance(pruned_data[k], dict)}
    for model in sorted(models):
        fs = full_data[model].get('score', 0.0)
        ps = pruned_data[model].get('score', 0.0)
        per_model[model] = {'full': round(fs, 4), 'pruned': round(ps, 4), 'delta': round(ps - fs, 4)}

    full_rank = sorted(per_model, key=lambda m: per_model[m]['full'], reverse=True)
    pruned_rank = sorted(per_model, key=lambda m: per_model[m]['pruned'], reverse=True)

    result = {
        'per_model': per_model,
        'rank_preserved': full_rank == pruned_rank,
        'pruned_count': pruned_data.get('__count__', '?'),
        'full_count': full_data.get('__count__', '?'),
        'warning': None,
    }
    _print_table('Comparison: Full vs Pruned', result)
    _print_warnings(result, 'comparison')


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Validate pruned benchmark subsets against full runs.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data-dir', metavar='DIR', help='Root directory containing predictions/ and reviews/ subdirs')
    parser.add_argument('--full', metavar='DIR', help='Directory with full-run result JSON files')
    parser.add_argument('--pruned', metavar='DIR', help='Directory with pruned-run result JSON files')
    args = parser.parse_args()

    if args.data_dir:
        run_data_dir(args.data_dir)
    elif args.full and args.pruned:
        run_dirs(args.full, args.pruned)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
