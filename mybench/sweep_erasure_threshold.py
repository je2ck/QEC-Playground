#!/usr/bin/env python3
"""
Sweep CSV → Hard Erasure Threshold Analysis

erasure_unsup_sweep.csv 파일에서 delta를 선택하여 hard erasure 시뮬레이션을 수행합니다.
_loss_cdf 열 (atom loss 확률 Bayes 포함) 을 사용합니다.

Parameters (from sweep CSV row at chosen delta):
  Pm = N_error / N_total                        (measurement error rate)
  Rm = P_erase_error_loss_cdf                   (P(erasure | error))
  Rc = P_erase_correct_loss_cdf                 (P(erasure | correct))

Usage:
    # List available deltas and their Rm/Rc:
    python sweep_erasure_threshold.py --sweep-csv erasure_unsup_sweep.csv --mode list

    # Show params for a specific delta:
    python sweep_erasure_threshold.py --sweep-csv erasure_unsup_sweep.csv --delta 0.495 --mode params

    # Quick simulation at one delta:
    python sweep_erasure_threshold.py --sweep-csv erasure_unsup_sweep.csv --delta 0.495 --mode quick

    # Sweep multiple deltas (compare erasure thresholds):
    python sweep_erasure_threshold.py --sweep-csv erasure_unsup_sweep.csv --delta-sweep 0.1 0.3 0.495 --mode quick

    # Full simulation:
    python sweep_erasure_threshold.py --sweep-csv erasure_unsup_sweep.csv --delta 0.495 --mode full

    # Search mode: fast sweep → find best delta → full simulation on best:
    python sweep_erasure_threshold.py --sweep-csv erasure_unsup_sweep.csv --mode search
    python sweep_erasure_threshold.py --sweep-csv erasure_unsup_sweep.csv --mode search --search-deltas 0.1 0.2 0.3 0.4 0.495
"""

import os
import sys
import json
import csv
import argparse
import subprocess

import numpy as np
import matplotlib.pyplot as plt

# QEC Playground path
qec_playground_root_dir = subprocess.run(
    "git rev-parse --show-toplevel",
    cwd=os.path.dirname(os.path.abspath(__file__)),
    shell=True, check=True, capture_output=True
).stdout.decode(sys.stdout.encoding).strip(" \r\n")
sys.path.insert(0, os.path.join(qec_playground_root_dir, "benchmark", "threshold_analyzer"))

from threshold_analyzer import (
    qecp_benchmark_simulate_func_command_vec,
    run_qecp_command_get_stdout,
    compile_code_if_necessary,
)
from utils import (run_p_sweep_with_checkpoint, clean_checkpoints,
    estimate_threshold_from_data,
    compute_lambda_factor, print_lambda_summary, plot_lambda_comparison,
    resolve_parallel_workers, save_lambda_results,
)


# ============== CSV Parsing ==============

def parse_sweep_csv(csv_path):
    """
    Parse erasure_unsup_sweep.csv.

    Returns:
        list of dicts, each row as floats (except empty strings → None).
    """
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                k = k.strip()
                v = v.strip()
                if v == '':
                    parsed[k] = None
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
                        pass
            rows.append(parsed)
    return rows


def find_row_by_delta(sweep_rows, delta):
    """Find the row closest to the requested delta."""
    best = None
    best_dist = float('inf')
    for row in sweep_rows:
        d = row.get('delta')
        if d is None:
            continue
        dist = abs(d - delta)
        if dist < best_dist:
            best_dist = dist
            best = row
    if best is None:
        raise ValueError(f"No valid delta found in sweep CSV")
    if best_dist > 1e-6:
        print(f"  [note] Requested delta={delta}, closest found: {best['delta']}")
    return best


def get_params_from_row(row, pm_override=None):
    """
    Extract Pm, Rm, Rc from a sweep CSV row (using _loss_cdf columns).

    Args:
        row: A parsed CSV row dict.
        pm_override: If given, use this value for Pm instead of N_error/N_total.
                     Use this when Rm/Rc are computed from a Gaussian posterior
                     whose implied Pm differs from the raw label error rate.

    Returns:
        (Pm, Rm, Rc, delta, N_erasure_loss)
    """
    if pm_override is not None:
        Pm = pm_override
    else:
        N_error = row['N_error']
        N_total = row['N_total']
        Pm = N_error / N_total if N_total > 0 else 0

    Rm = row.get('P_erase_error_loss_cdf', 0) or 0
    Rc = row.get('P_erase_correct_loss_cdf', 0) or 0
    delta = row.get('delta', 0)
    N_erasure_loss = row.get('N_erasure_loss', 0) or 0

    return Pm, Rm, Rc, delta, int(N_erasure_loss)


# ============== Display ==============

def print_sweep_table(sweep_rows, pm_override=None):
    """Print a summary table of all deltas."""
    if pm_override is not None:
        print(f"  (Using Pm override = {pm_override})")
    print(f"\n{'delta':>7} {'Pm':>8} {'Rm(loss)':>10} {'Rc(loss)':>12} {'ratio':>8} {'N_era':>7} {'era%':>7}")
    print('-' * 68)
    for row in sweep_rows:
        delta = row.get('delta')
        if delta is None:
            continue
        Pm, Rm, Rc, d, N_era = get_params_from_row(row, pm_override=pm_override)
        ratio = Rm / Rc if Rc > 0 else float('inf')
        era_frac = (Pm * Rm + (1 - Pm) * Rc) * 100
        print(f"{delta:>7.3f} {Pm:>8.5f} {Rm:>10.4%} {Rc:>12.6%} {ratio:>8.1f}x {N_era:>7} {era_frac:>6.3f}%")


def print_params_summary(Pm, Rm, Rc, delta, N_erasure_loss):
    """Print parameter summary for a chosen delta."""
    meas_err_rate = Pm * (1 - Rm)
    meas_err_with_erasure = Pm * Rm
    meas_erasure_no_error = (1 - Pm) * Rc
    total_erasure = meas_err_with_erasure + meas_erasure_no_error

    print(f"\n{'='*65}")
    print(f"  delta = {delta}  (loss model, hard erasure)")
    print(f"{'='*65}")
    print(f"  Pm  = {Pm:.6f}  (measurement error rate)")
    print(f"  Rm  = {Rm:.6f}  (P(erasure | error), loss_cdf)")
    print(f"  Rc  = {Rc:.8f}  (P(erasure | correct), loss_cdf)")
    if Rc > 0:
        print(f"  Rm/Rc ratio = {Rm/Rc:.1f}x")
    print(f"  N_erasure_loss = {N_erasure_loss}")
    print()
    print(f"  Derived noise model parameters:")
    print(f"    Hidden meas error:   Pm*(1-Rm) = {meas_err_rate:.6f}")
    print(f"    Erasure + error:     Pm*Rm     = {meas_err_with_erasure:.6f}")
    print(f"    Erasure only (FP):   (1-Pm)*Rc = {meas_erasure_no_error:.8f}")
    print(f"    Total erasure rate:             = {total_erasure:.6f}")
    if total_erasure > 0:
        p_err_given_erasure = meas_err_with_erasure / total_erasure
        print(f"    P(error | erasure):             = {p_err_given_erasure:.6f}")
    print(f"{'='*65}")


# ============== Simulate Functions ==============

def create_simulate_func_erasure(Pm, Rm, Rc):
    """Hard erasure simulation function."""
    measurement_error_rate = Pm * (1 - Rm)
    measurement_error_rate_with_erasure = Pm * Rm
    measurement_erasure_rate_no_error = (1 - Pm) * Rc

    def simulate_func(p, d, runtime_budget, p_graph=None):
        min_error_cases, time_budget = runtime_budget
        noisy_measurements = d

        config = {
            "use_correlated_pauli": True,
            "use_correlated_erasure": True,
            "measurement_error_rate": measurement_error_rate,
            "measurement_error_rate_with_erasure": measurement_error_rate_with_erasure,
            "measurement_erasure_rate_no_error": measurement_erasure_rate_no_error,
        }

        parameters = [
            "--code-type", "rotated-planar-code",
            "--noise-model-builder", "only-gate-error-circuit-level",
            "--noise-model-configuration", json.dumps(config),
            "--decoder", "union-find",
            "--decoder-config", '{"pcmg":true}',
        ]

        command = qecp_benchmark_simulate_func_command_vec(
            p, d, d, noisy_measurements, parameters,
            min_error_cases=min_error_cases,
            time_budget=time_budget,
            p_graph=p_graph,
        )

        stdout, returncode = run_qecp_command_get_stdout(command)
        if returncode != 0:
            print(f"  [ERROR] erasure sim failed for p={p}, d={d}")
            return (0.5, 1.0)

        full_result = stdout.strip(" \r\n").split("\n")[-1]
        lst = full_result.split(" ")
        pL = float(lst[5])
        pL_dev = float(lst[7])

        print(f"  [erasure] d={d:2d}, p={p:.4e}: pL={pL:.4e} ± {pL_dev:.2e}")
        return (pL, pL_dev)

    return simulate_func


def create_simulate_func_no_erasure(Pm):
    """No-erasure baseline."""
    def simulate_func(p, d, runtime_budget, p_graph=None):
        min_error_cases, time_budget = runtime_budget
        noisy_measurements = d

        config = {
            "use_correlated_pauli": True,
            "use_correlated_erasure": True,
            "measurement_error_rate": Pm,
        }

        parameters = [
            "--code-type", "rotated-planar-code",
            "--noise-model-builder", "only-gate-error-circuit-level",
            "--noise-model-configuration", json.dumps(config),
            "--decoder", "union-find",
            "--decoder-config", '{"pcmg":true}',
        ]

        command = qecp_benchmark_simulate_func_command_vec(
            p, d, d, noisy_measurements, parameters,
            min_error_cases=min_error_cases,
            time_budget=time_budget,
            p_graph=p_graph,
        )

        stdout, returncode = run_qecp_command_get_stdout(command)
        if returncode != 0:
            print(f"  [ERROR] no-erasure sim failed for p={p}, d={d}")
            return (0.5, 1.0)

        full_result = stdout.strip(" \r\n").split("\n")[-1]
        lst = full_result.split(" ")
        pL = float(lst[5])
        pL_dev = float(lst[7])

        print(f"  [no-erasure] d={d:2d}, p={p:.4e}: pL={pL:.4e} ± {pL_dev:.2e}")
        return (pL, pL_dev)

    return simulate_func


# ============== I/O ==============

def save_results(results, params, filename):
    """Save results to JSON."""
    data = {"params": params, "results": {}}
    for d, vals in results.items():
        data["results"][str(d)] = {
            "p": [float(x) for x in vals["p"]],
            "pL": [float(x) for x in vals["pL"]],
            "pL_dev": [float(x) for x in vals["pL_dev"]],
        }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {filename}")


def load_results(filename):
    """Load results from JSON."""
    with open(filename, 'r') as f:
        data = json.load(f)
    params = data.get("params", {})
    results = {}
    for d_str, vals in data["results"].items():
        results[int(d_str)] = vals
    return results, params


# ============== Plot ==============

def plot_single_delta(results_erasure, results_no_erasure, code_distances,
                      Pm, Rm, Rc, delta,
                      th_erasure=None, th_no=None,
                      save_path="sweep_threshold.pdf"):
    """Plot erasure vs no-erasure for a single delta."""
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {3: 'C0', 5: 'C1', 7: 'C2', 9: 'C3', 11: 'C4', 13: 'C5', 15: 'C6'}

    for d in code_distances:
        clr = colors.get(d, 'gray')

        if d in results_no_erasure and len(results_no_erasure[d]["p"]) > 0:
            p_arr = np.array(results_no_erasure[d]["p"])
            pL_arr = np.array(results_no_erasure[d]["pL"])
            valid = pL_arr > 0
            ax.plot(p_arr[valid], pL_arr[valid],
                    'o--', color=clr,
                    markerfacecolor='white', markeredgecolor=clr,
                    markersize=6, linewidth=1.5, alpha=0.8)

        if d in results_erasure and len(results_erasure[d]["p"]) > 0:
            p_arr = np.array(results_erasure[d]["p"])
            pL_arr = np.array(results_erasure[d]["pL"])
            valid = pL_arr > 0
            ax.plot(p_arr[valid], pL_arr[valid],
                    'o-', color=clr, markersize=6, linewidth=1.5,
                    label=f'd = {d}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical error probability, $p$', fontsize=14)
    ax.set_ylabel('Logical error rate, $p_L$', fontsize=14)

    all_p = []
    for d in code_distances:
        if d in results_erasure:
            all_p.extend(results_erasure[d]["p"])
        if d in results_no_erasure:
            all_p.extend(results_no_erasure[d]["p"])
    if all_p:
        ax.set_xlim(min(all_p) * 0.5, max(all_p) * 2)
    ax.set_ylim(1e-6, 1)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)

    y_pos = 1e-5
    if th_no is not None:
        ax.axvline(x=th_no, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        ax.text(th_no * 1.15, y_pos,
                f'$p_{{th}}={th_no*100:.2f}\\%$\n(no erasure)',
                fontsize=8, color='gray')
    if th_erasure is not None:
        ax.axvline(x=th_erasure, color='blue', linestyle='-', alpha=0.7, linewidth=2)
        ax.text(th_erasure * 0.35, y_pos,
                f'$p_{{th}}={th_erasure*100:.2f}\\%$\n(erasure)',
                fontsize=8, color='blue')

    ax.text(0.02, 0.18,
            'no erasure   (dashed, open)\n'
            'hard erasure  (solid, filled)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

    ratio = Rm / Rc if Rc > 0 else float('inf')
    title = (f"delta={delta}: Pm={Pm:.4f}, Rm={Rm:.4f}, Rc={Rc:.6f}, ratio={ratio:.0f}x")
    ax.set_title(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {save_path}")
    plt.show()


def plot_delta_sweep_summary(sweep_summary, save_path="sweep_delta_comparison.pdf"):
    """
    Plot Λ-factor or threshold vs delta for multiple delta runs.

    sweep_summary: list of {delta, Pm, Rm, Rc, lambda_data, threshold}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    deltas = [s['delta'] for s in sweep_summary]
    thresholds = [s.get('threshold') for s in sweep_summary]
    rms = [s['Rm'] for s in sweep_summary]

    # Left: threshold vs delta
    ax = axes[0]
    valid_th = [(d, t) for d, t in zip(deltas, thresholds) if t is not None]
    if valid_th:
        ds, ts = zip(*valid_th)
        ax.plot(ds, [t * 100 for t in ts], 'o-', color='C0', markersize=8)
    ax.set_xlabel('delta (soft_weight threshold)', fontsize=12)
    ax.set_ylabel('Threshold $p_{th}$ (%)', fontsize=12)
    ax.set_title('Threshold vs delta', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Right: Rm vs delta
    ax = axes[1]
    ax.plot(deltas, [r * 100 for r in rms], 's-', color='C1', markersize=8)
    ax.set_xlabel('delta (soft_weight threshold)', fontsize=12)
    ax.set_ylabel('Rm (%)', fontsize=12)
    ax.set_title('Erasure detection rate vs delta', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved delta sweep summary: {save_path}")
    plt.show()


# ============== Run Simulation ==============

def run_single_delta(sweep_rows, delta, code_distances, p_list, runtime_budget,
                     data_dir, n_workers=1, fresh=False, pm_override=None):
    """
    Run erasure + no-erasure simulation for a single delta.

    Returns:
        (results_erasure, results_no_erasure, Pm, Rm, Rc, delta_actual)
    """
    row = find_row_by_delta(sweep_rows, delta)
    Pm, Rm, Rc, delta_actual, N_era = get_params_from_row(row, pm_override=pm_override)
    print_params_summary(Pm, Rm, Rc, delta_actual, N_era)

    delta_tag = f"d{delta_actual:.3f}".replace('.', 'p')
    delta_dir = os.path.join(data_dir, delta_tag)
    os.makedirs(delta_dir, exist_ok=True)
    if fresh:
        clean_checkpoints(delta_dir)

    # 1) Erasure
    print(f"\n{'='*60}")
    print(f" [1/2] Hard erasure (delta={delta_actual}, Rm={Rm:.4f}, Rc={Rc:.6f})")
    print(f"{'='*60}")
    sim_era = create_simulate_func_erasure(Pm, Rm, Rc)
    results_era = run_p_sweep_with_checkpoint(
        sim_era, code_distances, p_list, runtime_budget,
        checkpoint_path=os.path.join(delta_dir, "checkpoint_erasure.json"),
        n_workers=n_workers)
    save_results(results_era,
                 {"delta": delta_actual, "Pm": Pm, "Rm": Rm, "Rc": Rc,
                  "type": "erasure"},
                 os.path.join(delta_dir, "results_erasure.json"))

    # 2) No erasure
    print(f"\n{'='*60}")
    print(f" [2/2] No erasure (Pm={Pm:.6f})")
    print(f"{'='*60}")
    sim_no = create_simulate_func_no_erasure(Pm)
    results_no = run_p_sweep_with_checkpoint(
        sim_no, code_distances, p_list, runtime_budget,
        checkpoint_path=os.path.join(delta_dir, "checkpoint_no_erasure.json"),
        n_workers=n_workers)
    save_results(results_no,
                 {"delta": delta_actual, "Pm": Pm, "type": "no_erasure"},
                 os.path.join(delta_dir, "results_no_erasure.json"))

    # Lambda
    lambda_era = compute_lambda_factor(results_era, code_distances)
    lambda_no = compute_lambda_factor(results_no, code_distances)
    print_lambda_summary(lambda_era, label=f"Erasure (delta={delta_actual})")
    print_lambda_summary(lambda_no, label="No erasure (baseline)")
    save_lambda_results(lambda_era, os.path.join(delta_dir, "lambda_erasure.json"))
    save_lambda_results(lambda_no, os.path.join(delta_dir, "lambda_no.json"))

    return results_era, results_no, Pm, Rm, Rc, delta_actual


# ============== Main ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sweep CSV → Hard Erasure Threshold (loss model)')
    parser.add_argument('--sweep-csv', required=True,
                        help='erasure_unsup_sweep.csv file path')
    parser.add_argument('--delta', type=float, default=None,
                        help='Single delta value to simulate')
    parser.add_argument('--delta-sweep', type=float, nargs='+', default=None,
                        help='Multiple delta values (e.g. --delta-sweep 0.1 0.3 0.495)')
    parser.add_argument('--mode', choices=['list', 'params', 'quick', 'full', 'plot', 'search'],
                        default='list',
                        help='Mode: list, params, quick, full, plot, search')
    parser.add_argument('--search-deltas', type=float, nargs='+', default=None,
                        help='Deltas to try in search mode (default: auto from CSV)')
    parser.add_argument('--search-n', type=int, default=8,
                        help='Number of deltas to auto-select in search mode (default: 8)')
    parser.add_argument('--p-range', type=float, nargs=2, default=None,
                        metavar=('LOG_MIN', 'LOG_MAX'),
                        help='p sweep range as log10 (default: -4 -1)')
    parser.add_argument('--n-points', type=int, default=None,
                        help='Number of p sweep points')
    parser.add_argument('--data-dir', default=None,
                        help='Directory for saving/loading result files')
    parser.add_argument('--output', default=None,
                        help='Output plot file path')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers (0 = all cores)')
    parser.add_argument('--fresh', action='store_true',
                        help='Delete existing checkpoints and start fresh')
    parser.add_argument('--pm-override', type=float, default=None,
                        help='Override Pm (measurement error rate). Use when Rm/Rc '
                             'are from Gaussian posterior with a different Pm than '
                             'the raw label rate in CSV. E.g. --pm-override 0.001')
    args = parser.parse_args()
    args.parallel = resolve_parallel_workers(args.parallel)
    pm_override = args.pm_override
    if pm_override is not None:
        print(f"[info] Using Pm override: {pm_override} (ignoring N_error/N_total from CSV)")

    # Parse sweep CSV
    sweep_rows = parse_sweep_csv(args.sweep_csv)

    if args.mode == 'list':
        print_sweep_table(sweep_rows, pm_override=pm_override)
        sys.exit(0)

    # Determine delta(s)
    deltas = []
    if args.mode == 'search':
        pass  # handled separately below
    elif args.delta_sweep:
        deltas = args.delta_sweep
    elif args.delta is not None:
        deltas = [args.delta]
    else:
        print("Error: specify --delta, --delta-sweep, or use --mode search")
        sys.exit(1)

    if args.mode == 'params':
        for delta in deltas:
            row = find_row_by_delta(sweep_rows, delta)
            Pm, Rm, Rc, d, N_era = get_params_from_row(row, pm_override=pm_override)
            print_params_summary(Pm, Rm, Rc, d, N_era)
        sys.exit(0)

    # Setup
    p_log_min, p_log_max = (args.p_range if args.p_range else [-4, -1])
    data_dir = args.data_dir or "results_sweep_erasure"
    os.makedirs(data_dir, exist_ok=True)

    compile_code_if_necessary()

    if args.mode in ('quick', 'full'):
        if args.mode == 'quick':
            code_distances = [5, 7, 9, 11]
            runtime_budget = (300, 45)
            n_default = 12
        else:
            code_distances = [3, 5, 7, 9, 11, 13]
            runtime_budget = (40000, 3600)
            n_default = 20

        n_points = args.n_points if args.n_points else n_default
        p_list = np.logspace(p_log_min, p_log_max, n_points).tolist()

        sweep_summary = []

        for delta in deltas:
            print(f"\n{'#'*70}")
            print(f"# DELTA = {delta}")
            print(f"{'#'*70}")

            results_era, results_no, Pm, Rm, Rc, delta_actual = run_single_delta(
                sweep_rows, delta, code_distances, p_list, runtime_budget,
                data_dir, n_workers=args.parallel, fresh=args.fresh,
                pm_override=pm_override)

            # Threshold estimate
            th_era, _ = estimate_threshold_from_data(results_era, code_distances, verbose=True)
            th_no, _ = estimate_threshold_from_data(results_no, code_distances, verbose=True)

            # Single-delta plot
            delta_tag = f"d{delta_actual:.3f}".replace('.', 'p')
            plot_path = args.output or os.path.join(
                data_dir, delta_tag, f"threshold_delta{delta_tag}.pdf")
            plot_single_delta(results_era, results_no, code_distances,
                              Pm, Rm, Rc, delta_actual,
                              th_erasure=th_era, th_no=th_no,
                              save_path=plot_path)

            sweep_summary.append({
                'delta': delta_actual,
                'Pm': Pm, 'Rm': Rm, 'Rc': Rc,
                'threshold': th_era,
                'threshold_no': th_no,
            })

        # Multi-delta summary
        if len(deltas) > 1:
            # Save summary
            summary_path = os.path.join(data_dir, "delta_sweep_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(sweep_summary, f, indent=2)
            print(f"\nSaved sweep summary: {summary_path}")

            # Summary plot
            summary_plot = os.path.join(data_dir, "delta_sweep_comparison.pdf")
            plot_delta_sweep_summary(sweep_summary, save_path=summary_plot)

            # Print table
            print(f"\n{'='*80}")
            print(f"  Delta Sweep Summary")
            print(f"{'='*80}")
            print(f"  {'delta':>7} {'Rm':>8} {'Rc':>10} {'ratio':>8} "
                  f"{'th_era':>10} {'th_no':>10} {'gain':>8}")
            print(f"  {'-'*70}")
            for s in sweep_summary:
                ratio = s['Rm'] / s['Rc'] if s['Rc'] > 0 else float('inf')
                th_e = f"{s['threshold']*100:.3f}%" if s['threshold'] else "N/A"
                th_n = f"{s['threshold_no']*100:.3f}%" if s['threshold_no'] else "N/A"
                gain = ""
                if s['threshold'] and s['threshold_no'] and s['threshold_no'] > 0:
                    gain = f"{s['threshold']/s['threshold_no']:.2f}x"
                print(f"  {s['delta']:>7.3f} {s['Rm']:>8.4f} {s['Rc']:>10.6f} "
                      f"{ratio:>8.1f}x {th_e:>10} {th_n:>10} {gain:>8}")
            print(f"{'='*80}")

    elif args.mode == 'search':
        # ============================================================
        # Search mode: fast sweep → find best delta → full simulation
        # ============================================================
        p_log_min_s, p_log_max_s = (args.p_range if args.p_range else [-4, -1])

        # Phase 1: determine search deltas
        if args.search_deltas:
            search_deltas = sorted(args.search_deltas)
        else:
            # Auto-select: pick N evenly-spaced deltas from CSV rows
            # Focus on the high-delta regime (0.46 ~ 0.498) where erasure is meaningful
            delta_lo, delta_hi = 0.46, 0.498
            available = sorted(set(
                row['delta'] for row in sweep_rows
                if row.get('delta') is not None and delta_lo <= row['delta'] <= delta_hi
            ))
            if len(available) <= args.search_n:
                search_deltas = available
            else:
                indices = np.linspace(0, len(available) - 1, args.search_n, dtype=int)
                search_deltas = [available[i] for i in indices]
            if not search_deltas:
                # Fallback: generate evenly-spaced deltas if CSV has none in range
                search_deltas = np.linspace(delta_lo, delta_hi, args.search_n).tolist()
        
        print(f"\n{'#'*70}")
        print(f"# SEARCH MODE: Phase 1 — Fast sweep over {len(search_deltas)} deltas")
        print(f"# Deltas: {[f'{d:.3f}' for d in search_deltas]}")
        print(f"{'#'*70}")

        # Phase 1: quick simulation for each delta
        fast_distances = [5, 7, 9, 11]
        fast_budget = (200, 30)
        fast_n_points = args.n_points if args.n_points else 10
        fast_p_list = np.logspace(p_log_min_s, p_log_max_s, fast_n_points).tolist()

        search_data_dir = os.path.join(data_dir, "search")
        os.makedirs(search_data_dir, exist_ok=True)
        if args.fresh:
            clean_checkpoints(search_data_dir)

        search_results = []
        for delta in search_deltas:
            print(f"\n{'='*60}")
            print(f" [search/fast] delta = {delta:.3f}")
            print(f"{'='*60}")

            row = find_row_by_delta(sweep_rows, delta)
            Pm, Rm, Rc, delta_actual, N_era = get_params_from_row(row, pm_override=pm_override)
            print(f"  Pm={Pm:.6f}, Rm={Rm:.4f}, Rc={Rc:.8f}")

            delta_tag = f"d{delta_actual:.3f}".replace('.', 'p')
            delta_dir = os.path.join(search_data_dir, delta_tag)
            os.makedirs(delta_dir, exist_ok=True)

            sim_era = create_simulate_func_erasure(Pm, Rm, Rc)
            results_era = run_p_sweep_with_checkpoint(
                sim_era, fast_distances, fast_p_list, fast_budget,
                checkpoint_path=os.path.join(delta_dir, "checkpoint_fast_erasure.json"),
                n_workers=args.parallel)

            th_era, _ = estimate_threshold_from_data(results_era, fast_distances, verbose=False)
            lambda_era = compute_lambda_factor(results_era, fast_distances)

            # Use median lambda as ranking metric (robust to outliers)
            # lambda_era is {(d_s, d_l): {"p": [...], "lambda": [...], ...}}
            lambda_vals = []
            for pair_data in lambda_era.values():
                lambda_vals.extend([v for v in pair_data["lambda"] if np.isfinite(v)])
            median_lambda = float(np.median(lambda_vals)) if lambda_vals else 0

            search_results.append({
                'delta': delta_actual,
                'Pm': Pm, 'Rm': Rm, 'Rc': Rc,
                'threshold': th_era,
                'median_lambda': median_lambda,
            })

            ratio = Rm / Rc if Rc > 0 else float('inf')
            th_str = f"{th_era*100:.3f}%" if th_era else "N/A"
            print(f"  → threshold={th_str}, median_Λ={median_lambda:.3f}, ratio={ratio:.0f}x")

        # Phase 1 summary
        print(f"\n{'='*70}")
        print(f"  Search Phase 1 — Fast Results")
        print(f"{'='*70}")
        print(f"  {'delta':>7} {'Rm':>8} {'Rc':>10} {'ratio':>8} {'threshold':>10} {'med_Λ':>8}")
        print(f"  {'-'*58}")
        for s in search_results:
            ratio = s['Rm'] / s['Rc'] if s['Rc'] > 0 else float('inf')
            th_str = f"{s['threshold']*100:.3f}%" if s['threshold'] else "N/A"
            print(f"  {s['delta']:>7.3f} {s['Rm']:>8.4f} {s['Rc']:>10.6f} "
                  f"{ratio:>8.0f}x {th_str:>10} {s['median_lambda']:>8.3f}")

        # Save search results
        search_summary_path = os.path.join(search_data_dir, "search_summary.json")
        with open(search_summary_path, 'w') as f:
            json.dump(search_results, f, indent=2)

        # Phase 2: pick the best delta → full simulation
        # Rank by threshold (higher is better), fall back to median_lambda
        valid_results = [s for s in search_results if s['threshold'] is not None]
        if not valid_results:
            print("\n  [WARNING] No valid thresholds found. Falling back to highest median_Λ.")
            valid_results = search_results
            best = max(valid_results, key=lambda s: s['median_lambda'])
        else:
            best = max(valid_results, key=lambda s: s['threshold'])

        best_delta = best['delta']
        print(f"\n{'#'*70}")
        print(f"# SEARCH MODE: Phase 2 — Full simulation at best delta = {best_delta:.3f}")
        print(f"# (threshold={best['threshold']*100:.3f}% from fast sweep)" if best['threshold'] else "")
        print(f"{'#'*70}")

        full_distances = [3, 5, 7, 9, 11, 13]
        full_budget = (40000, 3600)
        full_n_points = args.n_points if args.n_points else 20
        full_p_list = np.logspace(p_log_min_s, p_log_max_s, full_n_points).tolist()

        results_era, results_no, Pm, Rm, Rc, delta_actual = run_single_delta(
            sweep_rows, best_delta, full_distances, full_p_list, full_budget,
            data_dir, n_workers=args.parallel, fresh=False,
            pm_override=pm_override)

        th_era, _ = estimate_threshold_from_data(results_era, full_distances, verbose=True)
        th_no, _ = estimate_threshold_from_data(results_no, full_distances, verbose=True)

        delta_tag = f"d{delta_actual:.3f}".replace('.', 'p')
        plot_path = args.output or os.path.join(
            data_dir, delta_tag, f"threshold_delta{delta_tag}_search_best.pdf")
        plot_single_delta(results_era, results_no, full_distances,
                          Pm, Rm, Rc, delta_actual,
                          th_erasure=th_era, th_no=th_no,
                          save_path=plot_path)

        # Summary plot of search phase
        summary_plot = os.path.join(search_data_dir, "search_delta_comparison.pdf")
        plot_delta_sweep_summary(search_results, save_path=summary_plot)

        print(f"\n{'='*70}")
        print(f"  SEARCH COMPLETE")
        print(f"  Best delta: {best_delta:.3f} (Rm={Rm:.4f}, Rc={Rc:.6f})")
        if th_era:
            print(f"  Full threshold: {th_era*100:.3f}%")
        if th_no:
            print(f"  No-erasure baseline: {th_no*100:.3f}%")
        if th_era and th_no and th_no > 0:
            print(f"  Gain: {th_era/th_no:.2f}x")
        print(f"{'='*70}")

    elif args.mode == 'plot':
        for delta in deltas:
            row = find_row_by_delta(sweep_rows, delta)
            Pm, Rm, Rc, delta_actual, N_era = get_params_from_row(row, pm_override=pm_override)
            delta_tag = f"d{delta_actual:.3f}".replace('.', 'p')
            delta_dir = os.path.join(data_dir, delta_tag)

            results_era, _ = load_results(os.path.join(delta_dir, "results_erasure.json"))
            results_no, _ = load_results(os.path.join(delta_dir, "results_no_erasure.json"))
            code_distances = sorted(results_era.keys())

            th_era, _ = estimate_threshold_from_data(results_era, code_distances, verbose=True)
            th_no, _ = estimate_threshold_from_data(results_no, code_distances, verbose=True)

            plot_path = args.output or os.path.join(delta_dir, f"threshold_delta{delta_tag}.pdf")
            plot_single_delta(results_era, results_no, code_distances,
                              Pm, Rm, Rc, delta_actual,
                              th_erasure=th_era, th_no=th_no,
                              save_path=plot_path)
