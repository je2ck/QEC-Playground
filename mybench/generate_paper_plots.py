#!/usr/bin/env python3
"""
Generate all paper plots in one script.

6 plots:
  1. Raw vs Erasure(2d) threshold (no loss)
  2. Den vs Erasure(1d) vs Erasure(2d) (no loss)
  3. Raw vs Erasure(2d) threshold (realistic loss)
  4. Den vs Erasure(1d) vs Erasure(2d) (realistic loss)
  5. Rounds comparison: Raw vs Erasure(2d) (realistic loss)
  6. Lambda suppression: Raw vs Erasure(2d)

Usage:
    # All plots, quick mode
    python generate_paper_plots.py --mode quick --plots all --parallel 0

    # Single plot, full mode
    python generate_paper_plots.py --mode full --plots 1 --parallel 0

    # Replot from saved results
    python generate_paper_plots.py --mode plot --plots all
"""

import os
import sys
import json
import argparse
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# QEC Playground path
qec_playground_root_dir = subprocess.run(
    "git rev-parse --show-toplevel",
    cwd=os.path.dirname(os.path.abspath(__file__)),
    shell=True, check=True, capture_output=True
).stdout.decode(sys.stdout.encoding).strip(" \r\n")
sys.path.insert(0, os.path.join(qec_playground_root_dir, "benchmark", "threshold_analyzer"))

from threshold_analyzer import compile_code_if_necessary, ThresholdAnalyzer

from sweep_erasure_threshold import (
    parse_sweep_csv,
    find_row_by_delta,
    get_params_from_row,
    create_simulate_func_erasure,
    create_simulate_func_no_erasure,
    save_results as save_sweep_results,
    load_results as load_sweep_results,
)
from rounds_comparison_long import (
    build_rounds_list,
    run_rounds_sweep_parallel,
    run_rounds_sweep_sequential,
    make_noise_config_raw,
    make_noise_config_erasure,
    save_results as save_rounds_results,
    load_results as load_rounds_results,
    merge_checkpoint,
    filter_remaining_rounds,
)
from utils import (
    run_p_sweep_with_checkpoint,
    estimate_threshold_from_data,
    compute_lambda_factor,
    print_lambda_summary,
    scaled_runtime_budget,
    resolve_parallel_workers,
)

# ============== Constants ==============

MYBENCH_DIR = os.path.dirname(os.path.abspath(__file__))

PM_RAW = 0.023368
PM_DEN = 0.009313 
ATOM_LOSS = 0.00019 * 5  # 0.00095

CSV_2D = os.path.join(MYBENCH_DIR, "data", "5ms_erasure_unsup_sweep_2d.csv")
CSV_1D = os.path.join(MYBENCH_DIR, "data", "5ms_erasure_amp_sweep_1d.csv")
DELTA_2D = 0.475
DELTA_1D_DEFAULT = 0.15386374

CODE_DISTANCES_QUICK = [3, 5, 7, 9, 11]
CODE_DISTANCES_FULL = [3, 5, 7, 9, 11, 13]
# For threshold estimation: always use 5,7,9,11
CODE_DISTANCES_THRESHOLD = [5, 7, 9, 11]

RUNTIME_QUICK = (200, 60)
RUNTIME_FULL = (40000, 3600)
# Rough estimate uses fewer error cases and shorter time budget
RUNTIME_ROUGH_QUICK = (100, 30)
RUNTIME_ROUGH_FULL = (2000, 300)

ALL_PLOTS = [1, 2, 3, 4, 5, 6]


# ============== Helpers ==============

def get_erasure_params(csv_path, delta):
    """Load Rm, Rc from sweep CSV at given delta."""
    rows = parse_sweep_csv(csv_path)
    row = find_row_by_delta(rows, delta)
    _, Rm, Rc, delta_actual, _ = get_params_from_row(row)
    return Rm, Rc, delta_actual


def p_list_range(log_min, log_max, n_points):
    """Generate log-spaced p values."""
    return list(np.logspace(log_min, log_max, n_points))


def find_threshold_analyzer(simulate_func, code_distances, rough_runtime_budget,
                             runtime_budget, label="", rough_init_p=0.05):
    """
    ThresholdAnalyzer를 사용해 rough → precise threshold 추정.
    threshold_vs_measurement_error.py 방식 참고.

    Returns:
        threshold, threshold_err (None, None if failed)
    """
    rough_code_distances = [code_distances[0], code_distances[-1]]

    analyzer = ThresholdAnalyzer(
        code_distances=code_distances,
        simulate_func=simulate_func,
        default_rough_runtime_budget=rough_runtime_budget,
        default_runtime_budget=runtime_budget,
    )
    analyzer.rough_code_distances = rough_code_distances
    analyzer.rough_runtime_budgets = [rough_runtime_budget] * len(rough_code_distances)
    analyzer.verbose = True
    analyzer.rough_init_search_start_p = rough_init_p

    try:
        rough_popt, rough_perr = analyzer.rough_estimate()
        rough_th = rough_popt[3]
        print(f"  [{label}] Rough threshold: {rough_th:.6f} ({rough_th*100:.3f}%)")

        popt, perr = analyzer.precise_estimate(rough_popt)
        threshold = popt[3]
        threshold_err = perr[3]

        # Retry if error is too large
        if threshold_err / threshold > 0.01:
            print(f"  [{label}] Error too large ({threshold_err/threshold:.1%}), retrying...")
            popt, perr = analyzer.precise_estimate(popt)
            threshold = popt[3]
            threshold_err = perr[3]

        print(f"  [{label}] Threshold: {threshold:.6f} ({threshold*100:.3f}%) +/- {threshold_err:.6f}")
        return threshold, threshold_err

    except Exception as e:
        print(f"  [{label}] ThresholdAnalyzer failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to crossing-based method
        print(f"  [{label}] Falling back to crossing-based estimate...")
        th, th_err = estimate_threshold_from_data(
            analyzer.results if hasattr(analyzer, 'results') else {},
            code_distances, verbose=True)
        return th, th_err


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def clean_dir(path):
    """Remove all checkpoint/result files in a directory."""
    import glob as globmod
    for f in globmod.glob(os.path.join(path, "ckpt_*.json")):
        os.remove(f)
        print(f"  [fresh] Removed {os.path.basename(f)}")
    results_f = os.path.join(path, "results.json")
    if os.path.exists(results_f):
        os.remove(results_f)
        print(f"  [fresh] Removed results.json")


# ============== Plot Functions ==============

def plot_threshold_comparison(results_a, results_b, code_distances,
                               label_a, label_b, th_a, th_b,
                               title="", save_path="threshold.pdf"):
    """Plot two scenarios (each with multiple d) on one axes."""
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {3: 'C0', 5: 'C1', 7: 'C2', 9: 'C3', 11: 'C4', 13: 'C5', 15: 'C6'}

    for d in code_distances:
        clr = colors.get(d, 'gray')

        # Scenario A: dashed, open markers
        if d in results_a and len(results_a[d]["p"]) > 0:
            p_arr = np.array(results_a[d]["p"])
            pL_arr = np.array(results_a[d]["pL"])
            order = np.argsort(p_arr)
            p_arr, pL_arr = p_arr[order], pL_arr[order]
            valid = pL_arr > 0
            ax.plot(p_arr[valid], pL_arr[valid],
                    'o--', color=clr, markerfacecolor='white',
                    markeredgecolor=clr, markersize=6, linewidth=1.5, alpha=0.8)

        # Scenario B: solid, filled markers
        if d in results_b and len(results_b[d]["p"]) > 0:
            p_arr = np.array(results_b[d]["p"])
            pL_arr = np.array(results_b[d]["pL"])
            order = np.argsort(p_arr)
            p_arr, pL_arr = p_arr[order], pL_arr[order]
            valid = pL_arr > 0
            ax.plot(p_arr[valid], pL_arr[valid],
                    'o-', color=clr, markersize=6, linewidth=1.5,
                    label=f'd = {d}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical error probability, $p$', fontsize=14)
    ax.set_ylabel('Logical error rate, $p_L$', fontsize=14)
    ax.set_ylim(1e-6, 1)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)

    y_pos = 1e-5
    if th_a is not None:
        ax.axvline(x=th_a, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        ax.text(th_a * 1.15, y_pos,
                f'$p_{{th}}={th_a*100:.2f}\\%$\n({label_a})',
                fontsize=8, color='gray')
    if th_b is not None:
        ax.axvline(x=th_b, color='blue', linestyle='-', alpha=0.7, linewidth=2)
        ax.text(th_b * 0.35, y_pos,
                f'$p_{{th}}={th_b*100:.2f}\\%$\n({label_b})',
                fontsize=8, color='blue')

    legend_lines = []
    legend_lines.append(f'{label_a}  (dashed, open)')
    if th_a is not None:
        legend_lines[-1] += f'  $p_{{th}}={th_a*100:.2f}\\%$'
    else:
        legend_lines[-1] += '  (no threshold)'
    legend_lines.append(f'{label_b}  (solid, filled)')
    if th_b is not None:
        legend_lines[-1] += f'  $p_{{th}}={th_b*100:.2f}\\%$'
    else:
        legend_lines[-1] += '  (no threshold)'

    ax.text(0.02, 0.18,
            '\n'.join(legend_lines),
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

    if title:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_three_scenarios(results_dict, code_distances,
                          title="", save_path="comparison.pdf"):
    """
    Plot up to 3 scenarios overlaid.

    results_dict: {label: (results, color, linestyle, marker_fill)}
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors_d = {3: 'C0', 5: 'C1', 7: 'C2', 9: 'C3', 11: 'C4', 13: 'C5', 15: 'C6'}

    # For legend: one entry per scenario + one per d
    scenario_handles = []
    for label, (results, color, ls, fill) in results_dict.items():
        # Dummy line for scenario legend
        h, = ax.plot([], [], ls, color='black', linewidth=1.5, label=label)
        scenario_handles.append(h)

    d_handles = []
    for d in code_distances:
        clr = colors_d.get(d, 'gray')
        for label, (results, color, ls, fill) in results_dict.items():
            if d not in results or len(results[d]["p"]) == 0:
                continue
            p_arr = np.array(results[d]["p"])
            pL_arr = np.array(results[d]["pL"])
            order = np.argsort(p_arr)
            p_arr, pL_arr = p_arr[order], pL_arr[order]
            valid = pL_arr > 0
            mfc = clr if fill else 'white'
            ax.plot(p_arr[valid], pL_arr[valid],
                    marker='o', linestyle=ls.replace('o', ''),
                    color=clr, markerfacecolor=mfc, markeredgecolor=clr,
                    markersize=5, linewidth=1.5, alpha=0.85)
        # Dummy for d legend
        h, = ax.plot([], [], 'o', color=clr, markersize=6, label=f'd={d}')
        d_handles.append(h)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical error probability, $p$', fontsize=14)
    ax.set_ylabel('Logical error rate, $p_L$', fontsize=14)
    ax.set_ylim(1e-6, 1)

    legend1 = ax.legend(handles=scenario_handles, loc='lower right', fontsize=9, framealpha=0.9)
    ax.add_artist(legend1)
    ax.legend(handles=d_handles, loc='upper left', fontsize=9, framealpha=0.9)

    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    if title:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_rounds(all_results, code_distances, p_gate,
                Pm_raw, Pm_erasure, Rm, Rc, delta,
                pl_raw, pl_erasure,
                save_path="rounds.pdf"):
    """Plot per-round LER vs measurement rounds."""
    n_d = len(code_distances)
    fig, axes = plt.subplots(1, n_d, figsize=(5.5 * n_d, 5), squeeze=False)

    ratio = Rm / Rc if Rc > 0 else float('inf')
    styles = {
        'raw':     {'label': f'Raw (Pm={Pm_raw:.4f})',
                    'color': 'C3', 'marker': '^', 'ls': '--'},
        'erasure': {'label': f'Erasure (Pm={Pm_erasure:.4f}, Rm/Rc={ratio:.0f}x)',
                    'color': 'C0', 'marker': 'o', 'ls': '-'},
    }

    pl_values = {}
    if pl_raw > 0:
        pl_values['raw'] = pl_raw
    if pl_erasure > 0:
        pl_values['erasure'] = pl_erasure

    for i, d in enumerate(code_distances):
        ax = axes[0][i]
        n_ancillas = d * d - 1

        for sc, style in styles.items():
            if sc not in all_results or d not in all_results[sc]:
                continue
            data = all_results[sc][d]
            T_arr = np.array(data["T"], dtype=float)
            pL_arr = np.array(data["pL"])
            order = np.argsort(T_arr)
            T_arr, pL_arr = T_arr[order], pL_arr[order]
            valid = (pL_arr > 0) & (pL_arr < 1) & (T_arr > 0)
            if not np.any(valid):
                continue
            pL_per_round = 1 - (1 - pL_arr[valid]) ** (1.0 / T_arr[valid])
            ax.plot(T_arr[valid], pL_per_round,
                    marker=style['marker'], linestyle=style['ls'],
                    color=style['color'], markersize=4, linewidth=1.2,
                    label=style['label'])

        # Secondary y-axis: expected lost ancillas
        if pl_values:
            ax2 = ax.twinx()
            T_max = max(
                max(all_results[sc][d]["T"])
                for sc in all_results if d in all_results.get(sc, {})
            )
            T_range = np.linspace(d, T_max, 200)
            for sc_key, pl in pl_values.items():
                expected_lost = n_ancillas * (1 - (1 - pl) ** T_range)
                sc_color = 'C3' if sc_key == 'raw' else 'C0'
                ax2.plot(T_range, expected_lost,
                         color=sc_color, linestyle=':', alpha=0.4, linewidth=1.5)
            ax2.set_ylabel(f'E[lost ancillas] (of {n_ancillas})', fontsize=9, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray', labelsize=8)

        ax.axvline(x=d, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.set_yscale('log')
        ax.set_xlabel('Measurement rounds $T$', fontsize=12)
        ax.set_ylabel('Per-round logical error rate', fontsize=12)
        ax.set_title(f'd = {d}', fontsize=13)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Per-round LER vs Rounds (realistic loss, pl={pl_raw:.5f})\n'
                 f'p_gate={p_gate:.2e}, delta={delta:.3f}',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_lambda(lambda_datasets, code_distances_th,
                title="", save_path="lambda.pdf"):
    """Plot lambda suppression factor for multiple scenarios."""
    from utils import plot_lambda_comparison
    plot_lambda_comparison(lambda_datasets, code_distances_th, title=title, save_path=save_path)
    print(f"  Saved: {save_path}")


# ============== Simulation Runners ==============

def run_p_sweep(label, simulate_func, code_distances, p_list,
                runtime_budget, checkpoint_path, n_workers):
    """Run p-sweep with checkpoint support."""
    print(f"\n  [{label}] p-sweep: {len(code_distances)} distances x {len(p_list)} p-values")
    results = run_p_sweep_with_checkpoint(
        simulate_func, code_distances, p_list, runtime_budget,
        checkpoint_path=checkpoint_path, n_workers=n_workers,
    )
    return results


def run_rounds(label, p_gate, code_distances, rounds_list,
               noise_config, runtime_budget, n_workers,
               existing_results=None):
    """Run rounds sweep with checkpoint support."""
    if existing_results is None:
        existing_results = {}

    remaining, skipped = filter_remaining_rounds(rounds_list, existing_results)
    if not remaining:
        print(f"  [{label}] All rounds done (skipped {skipped})")
        return existing_results

    total = sum(len(v) for v in remaining.values())
    print(f"  [{label}] {total} round sims remaining (skipped {skipped})")

    remaining_distances = sorted(remaining.keys())
    from utils import ProgressTracker
    tracker = ProgressTracker(total, "sims", print_every=max(1, len(remaining_distances)))

    if n_workers > 1:
        new = run_rounds_sweep_parallel(
            label, p_gate, remaining_distances, remaining,
            noise_config, runtime_budget, n_workers, tracker=tracker)
    else:
        new = run_rounds_sweep_sequential(
            label, p_gate, remaining_distances, remaining,
            noise_config, runtime_budget, tracker=tracker)

    tracker.summary()
    return merge_checkpoint(existing_results, new)


# ============== Plot Implementations ==============

def do_plot1(output_dir, code_distances, code_distances_th,
             runtime_budget, n_workers, mode, rough_runtime_budget=None):
    """Plot 1: Raw vs Erasure(2d) threshold, no loss."""
    print(f"\n{'='*70}")
    print(f"  Plot 1: Raw vs Erasure(2d) threshold (no loss)")
    print(f"{'='*70}")

    data_dir = ensure_dir(os.path.join(output_dir, "plot1"))
    Rm, Rc, delta = get_erasure_params(CSV_2D, DELTA_2D)
    ratio = Rm / Rc if Rc > 0 else float('inf')
    print(f"  Pm_raw={PM_RAW:.6f}, Pm_den={PM_DEN:.6f}")
    print(f"  Rm={Rm:.6f}, Rc={Rc:.8f}, Rm/Rc={ratio:.0f}x, delta={delta:.3f}")

    p_list = p_list_range(-4, -1, 20 if mode == 'full' else 12)
    results_file = os.path.join(data_dir, "results.json")

    if mode == 'plot':
        if not os.path.exists(results_file):
            print(f"  [SKIP] No results file: {results_file}")
            return
        with open(results_file) as f:
            saved = json.load(f)
        results_raw = {int(k): v for k, v in saved["raw"].items()}
        results_era = {int(k): v for k, v in saved["erasure"].items()}
        th_raw = saved.get("th_raw")
        th_era = saved.get("th_erasure")
    else:
        sim_raw = create_simulate_func_no_erasure(PM_RAW)
        results_raw = run_p_sweep(
            "Raw", sim_raw, code_distances, p_list, runtime_budget,
            os.path.join(data_dir, "ckpt_raw.json"), n_workers)

        sim_era = create_simulate_func_erasure(PM_DEN, Rm, Rc)
        results_era = run_p_sweep(
            "Erasure(2d)", sim_era, code_distances, p_list, runtime_budget,
            os.path.join(data_dir, "ckpt_erasure.json"), n_workers)

        # Threshold estimation: rough → precise via ThresholdAnalyzer
        rough_budget = rough_runtime_budget or runtime_budget
        th_raw, _ = find_threshold_analyzer(
            sim_raw, code_distances_th, rough_budget, runtime_budget, label="Raw")
        th_era, _ = find_threshold_analyzer(
            sim_era, code_distances_th, rough_budget, runtime_budget, label="Erasure(2d)")

        saved = {
            "raw": {str(d): v for d, v in results_raw.items()},
            "erasure": {str(d): v for d, v in results_era.items()},
            "th_raw": th_raw, "th_erasure": th_era,
        }
        with open(results_file, 'w') as f:
            json.dump(saved, f, indent=2)

    plot_threshold_comparison(
        results_raw, results_era, code_distances,
        "Raw", "Erasure (2D)", th_raw, th_era,
        title=f"Raw (Pm={PM_RAW:.4f}) vs Erasure-2D (Pm={PM_DEN:.4f}, delta={delta:.3f})",
        save_path=os.path.join(data_dir, "plot1_threshold_raw_vs_erasure2d.pdf"),
    )
    return results_raw, results_era


def do_plot2(output_dir, code_distances, code_distances_th,
             runtime_budget, n_workers, mode, delta_1d=DELTA_1D_DEFAULT):
    """Plot 2: Den vs Erasure(1d) vs Erasure(2d), no loss."""
    print(f"\n{'='*70}")
    print(f"  Plot 2: Den vs Erasure(1d) vs Erasure(2d) (no loss)")
    print(f"{'='*70}")

    data_dir = ensure_dir(os.path.join(output_dir, "plot2"))
    Rm_2d, Rc_2d, delta_2d = get_erasure_params(CSV_2D, DELTA_2D)
    Rm_1d, Rc_1d, delta_1d = get_erasure_params(CSV_1D, delta_1d)
    print(f"  Pm_den={PM_DEN:.6f}")
    print(f"  2D: Rm={Rm_2d:.6f}, Rc={Rc_2d:.8f}, delta={delta_2d:.3f}")
    print(f"  1D: Rm={Rm_1d:.6f}, Rc={Rc_1d:.8f}, delta={delta_1d:.5f}")

    p_list = p_list_range(-3, -2, 20 if mode == 'full' else 12)
    results_file = os.path.join(data_dir, "results.json")

    if mode == 'plot':
        if not os.path.exists(results_file):
            print(f"  [SKIP] No results file")
            return
        with open(results_file) as f:
            saved = json.load(f)
        results_den = {int(k): v for k, v in saved["den"].items()}
        results_1d = {int(k): v for k, v in saved["erasure_1d"].items()}
        results_2d = {int(k): v for k, v in saved["erasure_2d"].items()}
    else:
        sim_den = create_simulate_func_no_erasure(PM_DEN)
        results_den = run_p_sweep(
            "Den", sim_den, code_distances, p_list, runtime_budget,
            os.path.join(data_dir, "ckpt_den.json"), n_workers)

        sim_1d = create_simulate_func_erasure(PM_DEN, Rm_1d, Rc_1d)
        results_1d = run_p_sweep(
            "Erasure(1d)", sim_1d, code_distances, p_list, runtime_budget,
            os.path.join(data_dir, "ckpt_1d.json"), n_workers)

        sim_2d = create_simulate_func_erasure(PM_DEN, Rm_2d, Rc_2d)
        results_2d = run_p_sweep(
            "Erasure(2d)", sim_2d, code_distances, p_list, runtime_budget,
            os.path.join(data_dir, "ckpt_2d.json"), n_workers)

        saved = {
            "den": {str(d): v for d, v in results_den.items()},
            "erasure_1d": {str(d): v for d, v in results_1d.items()},
            "erasure_2d": {str(d): v for d, v in results_2d.items()},
        }
        with open(results_file, 'w') as f:
            json.dump(saved, f, indent=2)

    plot_three_scenarios(
        {
            f'Den (Pm={PM_DEN:.4f})':
                (results_den, 'C3', '--', False),
            f'Erasure 1D (delta={delta_1d:.3f})':
                (results_1d, 'C1', '-.', True),
            f'Erasure 2D (delta={delta_2d:.3f})':
                (results_2d, 'C0', '-', True),
        },
        code_distances,
        title=f"Erasure Effect Comparison (Pm_den={PM_DEN:.4f})",
        save_path=os.path.join(data_dir, "plot2_erasure_comparison.pdf"),
    )


def do_plot3(output_dir, code_distances, code_distances_th,
             runtime_budget, n_workers, mode, rough_runtime_budget=None):
    """Plot 3: Raw vs Erasure(2d) threshold, realistic loss."""
    print(f"\n{'='*70}")
    print(f"  Plot 3: Raw vs Erasure(2d) threshold (realistic loss)")
    print(f"{'='*70}")

    data_dir = ensure_dir(os.path.join(output_dir, "plot3"))
    Rm, Rc, delta = get_erasure_params(CSV_2D, DELTA_2D)
    print(f"  Pm_raw={PM_RAW:.6f}, Pm_den={PM_DEN:.6f}, atom_loss={ATOM_LOSS:.5f}")

    p_list = p_list_range(-4, -1, 20 if mode == 'full' else 12)
    results_file = os.path.join(data_dir, "results.json")

    if mode == 'plot':
        if not os.path.exists(results_file):
            print(f"  [SKIP] No results file")
            return
        with open(results_file) as f:
            saved = json.load(f)
        results_raw = {int(k): v for k, v in saved["raw"].items()}
        results_era = {int(k): v for k, v in saved["erasure"].items()}
        th_raw = saved.get("th_raw")
        th_era = saved.get("th_erasure")
    else:
        sim_raw = create_simulate_func_no_erasure(PM_RAW, pl=ATOM_LOSS, realistic_loss=True)
        results_raw = run_p_sweep(
            "Raw+loss", sim_raw, code_distances, p_list, runtime_budget,
            os.path.join(data_dir, "ckpt_raw.json"), n_workers)

        sim_era = create_simulate_func_erasure(PM_DEN, Rm, Rc, pl=ATOM_LOSS, realistic_loss=True)
        results_era = run_p_sweep(
            "Erasure(2d)+loss", sim_era, code_distances, p_list, runtime_budget,
            os.path.join(data_dir, "ckpt_erasure.json"), n_workers)

        # Threshold estimation: rough → precise via ThresholdAnalyzer
        rough_budget = rough_runtime_budget or runtime_budget
        th_raw, _ = find_threshold_analyzer(
            sim_raw, code_distances_th, rough_budget, runtime_budget, label="Raw+loss")
        th_era, _ = find_threshold_analyzer(
            sim_era, code_distances_th, rough_budget, runtime_budget, label="Erasure(2d)+loss")

        saved = {
            "raw": {str(d): v for d, v in results_raw.items()},
            "erasure": {str(d): v for d, v in results_era.items()},
            "th_raw": th_raw, "th_erasure": th_era,
        }
        with open(results_file, 'w') as f:
            json.dump(saved, f, indent=2)

    plot_threshold_comparison(
        results_raw, results_era, code_distances,
        "Raw", "Erasure (2D)", th_raw, th_era,
        title=f"Raw vs Erasure-2D (realistic loss, pl={ATOM_LOSS:.5f})",
        save_path=os.path.join(data_dir, "plot3_threshold_with_loss.pdf"),
    )
    return results_raw, results_era


def do_plot4(output_dir, code_distances, code_distances_th,
             runtime_budget, n_workers, mode, delta_1d=DELTA_1D_DEFAULT):
    """Plot 4: Den vs Erasure(1d) vs Erasure(2d), realistic loss."""
    print(f"\n{'='*70}")
    print(f"  Plot 4: Den vs Erasure(1d) vs Erasure(2d) (realistic loss)")
    print(f"{'='*70}")

    data_dir = ensure_dir(os.path.join(output_dir, "plot4"))
    Rm_2d, Rc_2d, delta_2d = get_erasure_params(CSV_2D, DELTA_2D)
    Rm_1d, Rc_1d, delta_1d = get_erasure_params(CSV_1D, delta_1d)
    print(f"  Pm_den={PM_DEN:.6f}, atom_loss={ATOM_LOSS:.5f}")

    p_list = p_list_range(-3, -2, 20 if mode == 'full' else 12)
    results_file = os.path.join(data_dir, "results.json")

    if mode == 'plot':
        if not os.path.exists(results_file):
            print(f"  [SKIP] No results file")
            return
        with open(results_file) as f:
            saved = json.load(f)
        results_den = {int(k): v for k, v in saved["den"].items()}
        results_1d = {int(k): v for k, v in saved["erasure_1d"].items()}
        results_2d = {int(k): v for k, v in saved["erasure_2d"].items()}
    else:
        sim_den = create_simulate_func_no_erasure(PM_DEN, pl=ATOM_LOSS, realistic_loss=True)
        results_den = run_p_sweep(
            "Den+loss", sim_den, code_distances, p_list, runtime_budget,
            os.path.join(data_dir, "ckpt_den.json"), n_workers)

        sim_1d = create_simulate_func_erasure(PM_DEN, Rm_1d, Rc_1d, pl=ATOM_LOSS, realistic_loss=True)
        results_1d = run_p_sweep(
            "Era1d+loss", sim_1d, code_distances, p_list, runtime_budget,
            os.path.join(data_dir, "ckpt_1d.json"), n_workers)

        sim_2d = create_simulate_func_erasure(PM_DEN, Rm_2d, Rc_2d, pl=ATOM_LOSS, realistic_loss=True)
        results_2d = run_p_sweep(
            "Era2d+loss", sim_2d, code_distances, p_list, runtime_budget,
            os.path.join(data_dir, "ckpt_2d.json"), n_workers)

        saved = {
            "den": {str(d): v for d, v in results_den.items()},
            "erasure_1d": {str(d): v for d, v in results_1d.items()},
            "erasure_2d": {str(d): v for d, v in results_2d.items()},
        }
        with open(results_file, 'w') as f:
            json.dump(saved, f, indent=2)

    plot_three_scenarios(
        {
            f'Den (Pm={PM_DEN:.4f})':
                (results_den, 'C3', '--', False),
            f'Erasure 1D (delta={delta_1d:.3f})':
                (results_1d, 'C1', '-.', True),
            f'Erasure 2D (delta={delta_2d:.3f})':
                (results_2d, 'C0', '-', True),
        },
        code_distances,
        title=f"Erasure Comparison (realistic loss, pl={ATOM_LOSS:.5f})",
        save_path=os.path.join(data_dir, "plot4_erasure_comparison_with_loss.pdf"),
    )


def do_plot5(output_dir, code_distances, runtime_budget, n_workers, mode,
             p_gate, max_rounds):
    """Plot 5: Rounds comparison Raw vs Erasure(2d), realistic loss."""
    print(f"\n{'='*70}")
    print(f"  Plot 5: Rounds comparison (realistic loss)")
    print(f"{'='*70}")

    data_dir = ensure_dir(os.path.join(output_dir, "plot5"))
    Rm, Rc, delta = get_erasure_params(CSV_2D, DELTA_2D)
    print(f"  p_gate={p_gate:.2e}, pl={ATOM_LOSS:.5f}, max_rounds={max_rounds}")

    results_file = os.path.join(data_dir, "results.json")

    if mode == 'plot':
        if not os.path.exists(results_file):
            print(f"  [SKIP] No results file")
            return
        all_results, _ = load_rounds_results(results_file)
    else:
        # Build per-d rounds list
        rounds_list = {}
        for d in code_distances:
            rounds_list[d] = build_rounds_list(d, max_rounds)

        for d in code_distances:
            rlist = rounds_list[d]
            print(f"    d={d}: {len(rlist)} points, T={rlist[0]}..{rlist[-1]}")

        # Load existing
        all_results = {}
        if os.path.exists(results_file):
            all_results, _ = load_rounds_results(results_file)

        noise_raw = make_noise_config_raw(PM_RAW, pl=ATOM_LOSS, realistic_loss=True)
        noise_era = make_noise_config_erasure(PM_DEN, Rm, Rc, pl=ATOM_LOSS, realistic_loss=True)

        scenarios = [
            ('raw', 'Raw', noise_raw),
            ('erasure', 'Erasure', noise_era),
        ]

        for sc_key, sc_label, noise_config in scenarios:
            existing = all_results.get(sc_key, {})
            all_results[sc_key] = run_rounds(
                sc_label, p_gate, code_distances, rounds_list,
                noise_config, runtime_budget, n_workers,
                existing_results=existing,
            )
            save_rounds_results(all_results, {
                "p_gate": p_gate, "pl": ATOM_LOSS,
                "delta": delta, "max_rounds": max_rounds,
            }, results_file)

    plot_rounds(
        all_results, code_distances, p_gate,
        PM_RAW, PM_DEN, Rm, Rc, delta,
        ATOM_LOSS, ATOM_LOSS,
        save_path=os.path.join(data_dir, "plot5_rounds_comparison.pdf"),
    )


def do_plot6(output_dir, code_distances, code_distances_th,
             runtime_budget, n_workers, mode,
             plot1_results=None, plot3_results=None):
    """Plot 6: Lambda suppression Raw vs Erasure(2d), no-loss and with-loss."""
    print(f"\n{'='*70}")
    print(f"  Plot 6: Lambda suppression (no loss + realistic loss)")
    print(f"{'='*70}")

    data_dir = ensure_dir(os.path.join(output_dir, "plot6"))
    Rm, Rc, delta = get_erasure_params(CSV_2D, DELTA_2D)

    p_list = p_list_range(-4, -1, 20 if mode == 'full' else 12)
    results_file = os.path.join(data_dir, "results.json")

    def load_or_run(key, fallback_file, fallback_results, sim_func, label, ckpt_suffix):
        """Try: passed results → saved file → fallback plot file → run fresh."""
        if fallback_results is not None:
            return fallback_results
        if os.path.exists(results_file):
            with open(results_file) as f:
                saved = json.load(f)
            if key in saved:
                return {int(k): v for k, v in saved[key].items()}
        if fallback_file and os.path.exists(fallback_file):
            with open(fallback_file) as f:
                saved = json.load(f)
            # plot1/plot3 results have "raw" and "erasure" keys
            src_key = "raw" if "raw" in key else "erasure"
            if src_key in saved:
                return {int(k): v for k, v in saved[src_key].items()}
        if mode == 'plot':
            return None
        return run_p_sweep(
            label, sim_func, code_distances, p_list, runtime_budget,
            os.path.join(data_dir, f"ckpt_{ckpt_suffix}.json"), n_workers)

    plot1_file = os.path.join(output_dir, "plot1", "results.json")
    plot3_file = os.path.join(output_dir, "plot3", "results.json")

    # --- No loss ---
    p1_raw = plot1_results[0] if plot1_results else None
    p1_era = plot1_results[1] if plot1_results else None

    results_raw_noloss = load_or_run(
        "raw_noloss", plot1_file, p1_raw,
        create_simulate_func_no_erasure(PM_RAW),
        "Raw(no loss)", "raw_noloss")
    results_era_noloss = load_or_run(
        "era_noloss", plot1_file, p1_era,
        create_simulate_func_erasure(PM_DEN, Rm, Rc),
        "Era(no loss)", "era_noloss")

    # --- With loss ---
    p3_raw = plot3_results[0] if plot3_results else None
    p3_era = plot3_results[1] if plot3_results else None

    results_raw_loss = load_or_run(
        "raw_loss", plot3_file, p3_raw,
        create_simulate_func_no_erasure(PM_RAW, pl=ATOM_LOSS, realistic_loss=True),
        "Raw(loss)", "raw_loss")
    results_era_loss = load_or_run(
        "era_loss", plot3_file, p3_era,
        create_simulate_func_erasure(PM_DEN, Rm, Rc, pl=ATOM_LOSS, realistic_loss=True),
        "Era(loss)", "era_loss")

    # Check if any results missing
    all_sets = {
        "raw_noloss": results_raw_noloss,
        "era_noloss": results_era_noloss,
        "raw_loss": results_raw_loss,
        "era_loss": results_era_loss,
    }
    missing = [k for k, v in all_sets.items() if v is None]
    if missing:
        print(f"  [SKIP] Missing results for: {missing}")
        print(f"         Run plots 1 and 3 first.")
        return

    # Save all for reuse
    saved = {}
    for key, res in all_sets.items():
        saved[key] = {str(d): v for d, v in res.items()}
    with open(results_file, 'w') as f:
        json.dump(saved, f, indent=2)

    # Compute lambda for all 4 scenarios
    lambda_raw_noloss = compute_lambda_factor(results_raw_noloss, code_distances_th)
    lambda_era_noloss = compute_lambda_factor(results_era_noloss, code_distances_th)
    lambda_raw_loss = compute_lambda_factor(results_raw_loss, code_distances_th)
    lambda_era_loss = compute_lambda_factor(results_era_loss, code_distances_th)

    print_lambda_summary(lambda_raw_noloss, "Raw (no loss)")
    print_lambda_summary(lambda_era_noloss, "Erasure 2D (no loss)")
    print_lambda_summary(lambda_raw_loss, "Raw (realistic loss)")
    print_lambda_summary(lambda_era_loss, "Erasure 2D (realistic loss)")

    # Save lambda
    def serialize_lambda(ld):
        return {f"{k[0]}_{k[1]}": v for k, v in ld.items()}

    with open(os.path.join(data_dir, "lambda.json"), 'w') as f:
        json.dump({
            "raw_noloss": serialize_lambda(lambda_raw_noloss),
            "era_noloss": serialize_lambda(lambda_era_noloss),
            "raw_loss": serialize_lambda(lambda_raw_loss),
            "era_loss": serialize_lambda(lambda_era_loss),
        }, f, indent=2)

    # Plot all 4 on one figure
    datasets = [
        ('Raw (no loss)',               'C3', '^', '--', lambda_raw_noloss),
        ('Erasure 2D (no loss)',        'C0', 'o', '-',  lambda_era_noloss),
        (f'Raw (loss, pl={ATOM_LOSS:.4f})',      'C1', 's', '--', lambda_raw_loss),
        (f'Erasure 2D (loss, pl={ATOM_LOSS:.4f})', 'C2', 'D', '-',  lambda_era_loss),
    ]
    plot_lambda(
        datasets, code_distances_th,
        title=f"Lambda Suppression: Raw vs Erasure-2D (no loss & realistic loss)",
        save_path=os.path.join(data_dir, "plot6_lambda_suppression.pdf"),
    )


# ============== Main ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate all paper plots')
    parser.add_argument('--mode', choices=['quick', 'full', 'plot'], required=True,
                        help='quick: fast test, full: paper quality, plot: replot only')
    parser.add_argument('--plots', nargs='+', default=['all'],
                        help='Plot numbers to generate (e.g. --plots 1 3 6 or --plots all)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Parallel workers (0=all cores, 1=sequential)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory (default: results_paper)')
    parser.add_argument('--p-gate', type=float, default=0.0005,
                        help='Gate error rate for plot 5 (default: 0.0005)')
    parser.add_argument('--max-rounds', type=int, default=100,
                        help='Max measurement rounds for plot 5 (default: 100)')
    parser.add_argument('--delta-1d', type=float, default=DELTA_1D_DEFAULT,
                        help=f'Delta for 1D erasure (default: {DELTA_1D_DEFAULT})')
    parser.add_argument('--fresh', action='store_true',
                        help='Delete existing checkpoints and start fresh')
    args = parser.parse_args()

    n_workers = resolve_parallel_workers(args.parallel)
    output_dir = ensure_dir(
        args.output_dir or os.path.join(MYBENCH_DIR, "results_paper"))

    # Parse --plots
    if 'all' in args.plots:
        plots_to_run = ALL_PLOTS
    else:
        plots_to_run = sorted(set(int(x) for x in args.plots))
        for p in plots_to_run:
            if p not in ALL_PLOTS:
                print(f"Error: invalid plot number {p}. Valid: {ALL_PLOTS}")
                sys.exit(1)

    # Mode-dependent settings
    code_distances_th = CODE_DISTANCES_THRESHOLD  # always [5, 7, 9, 11]
    if args.mode == 'quick':
        code_distances = CODE_DISTANCES_QUICK
        runtime_budget = RUNTIME_QUICK
        rough_runtime_budget = RUNTIME_ROUGH_QUICK
    elif args.mode == 'full':
        code_distances = CODE_DISTANCES_FULL
        runtime_budget = RUNTIME_FULL
        rough_runtime_budget = RUNTIME_ROUGH_FULL
    else:
        # plot mode: distances will be inferred from saved data
        code_distances = CODE_DISTANCES_FULL
        runtime_budget = None
        rough_runtime_budget = None

    print(f"\n{'#'*70}")
    print(f"  Paper Plot Generator")
    print(f"  Mode: {args.mode}, Plots: {plots_to_run}")
    print(f"  Code distances: {code_distances}")
    print(f"  Threshold distances: {code_distances_th}")
    print(f"  Output: {output_dir}")
    print(f"  Workers: {n_workers}")
    print(f"{'#'*70}")

    # Compile simulator if needed
    if args.mode != 'plot':
        compile_code_if_necessary()

    # Clean checkpoints if --fresh
    if args.fresh and args.mode != 'plot':
        for p in plots_to_run:
            plot_dir = os.path.join(output_dir, f"plot{p}")
            if os.path.isdir(plot_dir):
                clean_dir(plot_dir)

    # Run selected plots
    plot1_results = None
    plot3_results = None

    if 1 in plots_to_run:
        ret = do_plot1(output_dir, code_distances, code_distances_th,
                       runtime_budget, n_workers, args.mode,
                       rough_runtime_budget=rough_runtime_budget)
        if ret is not None:
            plot1_results = ret

    if 2 in plots_to_run:
        do_plot2(output_dir, code_distances, code_distances_th,
                 runtime_budget, n_workers, args.mode,
                 delta_1d=args.delta_1d)

    if 3 in plots_to_run:
        ret = do_plot3(output_dir, code_distances, code_distances_th,
                       runtime_budget, n_workers, args.mode,
                       rough_runtime_budget=rough_runtime_budget)
        if ret is not None:
            plot3_results = ret

    if 4 in plots_to_run:
        do_plot4(output_dir, code_distances, code_distances_th,
                 runtime_budget, n_workers, args.mode,
                 delta_1d=args.delta_1d)

    if 5 in plots_to_run:
        do_plot5(output_dir, code_distances, runtime_budget, n_workers,
                 args.mode, args.p_gate, args.max_rounds)

    if 6 in plots_to_run:
        do_plot6(output_dir, code_distances, code_distances_th,
                 runtime_budget, n_workers, args.mode,
                 plot1_results=plot1_results,
                 plot3_results=plot3_results)

    print(f"\n{'#'*70}")
    print(f"  Done! All outputs in: {output_dir}")
    print(f"{'#'*70}")
