#!/usr/bin/env python3
"""
Long-range Rounds Comparison: Raw vs Erasure

고정된 gate error p와 code distance d에서 measurement rounds T를 d ~ max_rounds까지
늘려가며 per-round LER을 비교합니다.

Two scenarios:
  1. Raw:     Pm_raw, pl_raw (no erasure)
  2. Erasure: Pm_erasure, pl_erasure + hard erasure (Rm, Rc from sweep CSV at delta)

X-axis: measurement rounds T
Y-axis: per-round logical error rate

Usage:
    # fast (Pm과 atom loss를 각각 지정)
    python rounds_comparison_long.py --sweep-csv erasure_unsup_sweep.csv \
        --pm-raw 0.01 --pm-erasure 0.008 \
        --pl-raw 0.001 --pl-erasure 0.001 \
        --mode fast --p-gate 0.001

    # full simulation
    python rounds_comparison_long.py --sweep-csv erasure_unsup_sweep.csv \
        --pm-raw 0.01 --pm-erasure 0.008 --mode full --p-gate 0.001

    # parallel
    python rounds_comparison_long.py --sweep-csv erasure_unsup_sweep.csv \
        --pm-raw 0.01 --pm-erasure 0.008 --mode full --parallel 0

    # custom delta & max rounds
    python rounds_comparison_long.py --sweep-csv erasure_unsup_sweep.csv \
        --pm-raw 0.01 --pm-erasure 0.008 --delta 0.49 --max-rounds 200 --mode full

    # refresh (discard previous results)
    python rounds_comparison_long.py --sweep-csv erasure_unsup_sweep.csv \
        --pm-raw 0.01 --pm-erasure 0.008 --mode full --refresh

    # plot only (from saved results)
    python rounds_comparison_long.py --sweep-csv erasure_unsup_sweep.csv --mode plot
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
from utils import ProgressTracker, scaled_runtime_budget, resolve_parallel_workers


# ============== CSV Parsing (from sweep_erasure_threshold.py) ==============

def parse_sweep_csv(csv_path):
    """Parse erasure_unsup_sweep.csv."""
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
        raise ValueError("No valid delta found in sweep CSV")
    if best_dist > 1e-6:
        print(f"  [note] Requested delta={delta}, closest found: {best['delta']}")
    return best


def get_params_from_row(row, pm_override=None):
    """Extract Pm, Rm, Rc from a sweep CSV row (using _loss_cdf columns)."""
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


# ============== Simulate ==============

def run_single_simulation(p, d, T, noise_config, runtime_budget):
    """Run a single QEC simulation."""
    min_error_cases, time_budget = runtime_budget

    parameters = [
        "--code-type", "rotated-planar-code",
        "--noise-model-builder", "only-gate-error-circuit-level",
        "--noise-model-configuration", json.dumps(noise_config),
        "--decoder", "union-find",
        "--decoder-config", '{"pcmg":true}',
    ]

    command = qecp_benchmark_simulate_func_command_vec(
        p, d, d, T, parameters,
        min_error_cases=min_error_cases,
        time_budget=time_budget,
        p_graph=p,
    )

    stdout, returncode = run_qecp_command_get_stdout(command)
    if returncode != 0:
        return (0.5, 1.0)

    full_result = stdout.strip(" \r\n").split("\n")[-1]
    lst = full_result.split(" ")
    pL = float(lst[5])
    pL_dev = float(lst[7])

    return (pL, pL_dev)


def make_noise_config_raw(Pm, pl=0.0):
    """Raw: measurement_error_rate = Pm, no erasure, optional atom loss."""
    config = {
        "use_correlated_pauli": True,
        "use_correlated_erasure": True,
        "measurement_error_rate": Pm,
    }
    if pl > 0:
        config["ancilla_loss_probability"] = pl
    return config


def make_noise_config_erasure(Pm, Rm, Rc, pl=0.0):
    """Erasure: hard erasure model, optional atom loss."""
    measurement_error_rate = Pm * (1 - Rm)
    measurement_error_rate_with_erasure = Pm * Rm
    measurement_erasure_rate_no_error = (1 - Pm) * Rc
    config = {
        "use_correlated_pauli": True,
        "use_correlated_erasure": True,
        "measurement_error_rate": measurement_error_rate,
        "measurement_error_rate_with_erasure": measurement_error_rate_with_erasure,
        "measurement_erasure_rate_no_error": measurement_erasure_rate_no_error,
    }
    if pl > 0:
        config["ancilla_loss_probability"] = pl
    return config


# ============== Rounds Sweep ==============

def build_rounds_list(d, max_rounds, step=None):
    """Build list of rounds from d to max_rounds."""
    if step is None:
        # Adaptive: sparse at large T
        rounds = list(range(d, min(d + 20, max_rounds + 1), 2))
        if d + 20 < max_rounds:
            rounds += list(range(d + 20, min(d + 60, max_rounds + 1), 5))
        if d + 60 < max_rounds:
            rounds += list(range(d + 60, max_rounds + 1, 10))
        # Ensure max_rounds is included
        if max_rounds not in rounds:
            rounds.append(max_rounds)
        return sorted(set(rounds))
    else:
        rounds = list(range(d, max_rounds + 1, step))
        if max_rounds not in rounds:
            rounds.append(max_rounds)
        return rounds


def run_rounds_sweep_parallel(label, p_gate, code_distances, rounds_list,
                              noise_config, runtime_budget,
                              n_workers, tracker=None):
    """Parallel sweep: all (d, T) pairs submitted concurrently."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = [(d, T) for d in code_distances for T in rounds_list[d]]
    print(f"  \u26a1 [{label}] Parallel: {n_workers} workers, {len(tasks)} tasks")

    d_base = min(code_distances)
    result_map = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_key = {}
        for d, T in tasks:
            budget = scaled_runtime_budget(runtime_budget, d, d_base)
            future = executor.submit(
                run_single_simulation, p_gate, d, T,
                noise_config, budget
            )
            future_to_key[future] = (d, T)

        for future in as_completed(future_to_key):
            d, T = future_to_key[future]
            try:
                pL, pL_dev = future.result()
            except Exception as e:
                print(f"  [ERROR] [{label}] d={d}, T={T}: {e}")
                pL, pL_dev = 0.5, 1.0
            result_map[(d, T)] = (pL, pL_dev)
            print(f"  [{label}] d={d:2d}, T={T:3d}: pL={pL:.4e} \u00b1 {pL_dev:.2e}")
            if tracker:
                tracker.task_done()

    # Organize by code distance
    results = {}
    for d in code_distances:
        results[d] = {"T": [], "pL": [], "pL_dev": []}
        for T in rounds_list[d]:
            pL, pL_dev = result_map[(d, T)]
            results[d]["T"].append(T)
            results[d]["pL"].append(pL)
            results[d]["pL_dev"].append(pL_dev)

    return results


def run_rounds_sweep_sequential(label, p_gate, code_distances, rounds_list,
                                noise_config, runtime_budget, tracker=None):
    """Sequential sweep."""
    d_base = min(code_distances)
    results = {}

    for d in code_distances:
        results[d] = {"T": [], "pL": [], "pL_dev": []}
        print(f"\n  --- d = {d} ---")
        budget = scaled_runtime_budget(runtime_budget, d, d_base)
        for T in rounds_list[d]:
            if tracker:
                tracker.begin_task()
            pL, pL_dev = run_single_simulation(
                p_gate, d, T, noise_config, budget
            )
            print(f"  [{label}] d={d:2d}, T={T:3d}: pL={pL:.4e} \u00b1 {pL_dev:.2e}")
            results[d]["T"].append(T)
            results[d]["pL"].append(pL)
            results[d]["pL_dev"].append(pL_dev)
            if tracker:
                tracker.end_task()

    return results


# ============== Checkpoint I/O ==============

def save_results(all_results, params, filename):
    """Save results to JSON."""
    data = {"params": params, "results": {}}
    for scenario, d_results in all_results.items():
        data["results"][scenario] = {}
        for d, vals in d_results.items():
            data["results"][scenario][str(d)] = {
                "T": vals["T"],
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
    all_results = {}
    for scenario, d_dict in data["results"].items():
        all_results[scenario] = {}
        for d_str, vals in d_dict.items():
            all_results[scenario][int(d_str)] = vals
    return all_results, params


def merge_checkpoint(existing, new_data):
    """Merge new simulation results into existing, avoiding duplicates."""
    for d in new_data:
        if d not in existing:
            existing[d] = new_data[d]
            continue
        existing_T_set = set(existing[d]["T"])
        for i, T in enumerate(new_data[d]["T"]):
            if T not in existing_T_set:
                existing[d]["T"].append(T)
                existing[d]["pL"].append(new_data[d]["pL"][i])
                existing[d]["pL_dev"].append(new_data[d]["pL_dev"][i])
        # Sort by T
        if existing[d]["T"]:
            order = np.argsort(existing[d]["T"])
            existing[d]["T"] = [existing[d]["T"][i] for i in order]
            existing[d]["pL"] = [existing[d]["pL"][i] for i in order]
            existing[d]["pL_dev"] = [existing[d]["pL_dev"][i] for i in order]
    return existing


def filter_remaining_rounds(rounds_list, existing_results):
    """Remove already-completed (d, T) pairs from rounds_list."""
    filtered = {}
    skipped = 0
    for d, T_list in rounds_list.items():
        if d in existing_results:
            done = set(existing_results[d]["T"])
            remaining = [T for T in T_list if T not in done]
            skipped += len(T_list) - len(remaining)
        else:
            remaining = list(T_list)
        if remaining:
            filtered[d] = remaining
    return filtered, skipped


# ============== Plot ==============

def plot_rounds_comparison(all_results, code_distances, p_gate, delta,
                           Pm_raw, Pm_erasure, Rm, Rc,
                           save_path="rounds_comparison_long.pdf"):
    """Plot per-round LER vs measurement rounds T for Raw vs Erasure."""
    n_d = len(code_distances)
    fig, axes = plt.subplots(1, n_d, figsize=(5.5 * n_d, 5), squeeze=False)

    ratio = Rm / Rc if Rc > 0 else float('inf')
    scenario_styles = {
        'raw':     {'label': f'Raw (Pm={Pm_raw:.4f})',
                    'color': 'C3', 'marker': '^', 'ls': '--'},
        'erasure': {'label': f'Erasure (Pm={Pm_erasure:.4f}, \u03b4={delta:.3f}, Rm/Rc={ratio:.0f}x)',
                    'color': 'C0', 'marker': 'o', 'ls': '-'},
    }

    for i, d in enumerate(code_distances):
        ax = axes[0][i]

        for scenario, style in scenario_styles.items():
            if scenario not in all_results or d not in all_results[scenario]:
                continue
            data = all_results[scenario][d]
            T_arr = np.array(data["T"], dtype=float)
            pL_arr = np.array(data["pL"])

            valid = (pL_arr > 0) & (pL_arr < 1) & (T_arr > 0)
            if not np.any(valid):
                continue

            pL_per_round = 1 - (1 - pL_arr[valid]) ** (1.0 / T_arr[valid])

            ax.plot(T_arr[valid], pL_per_round,
                    marker=style['marker'], linestyle=style['ls'],
                    color=style['color'], markersize=4, linewidth=1.2,
                    label=style['label'])

        ax.axvline(x=d, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(d + 0.5, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-6,
                f'T=d={d}', fontsize=7, color='gray', rotation=90, va='bottom')

        ax.set_yscale('log')
        ax.set_xlabel('Measurement rounds $T$', fontsize=12)
        ax.set_ylabel('Per-round logical error rate', fontsize=12)
        ax.set_title(f'd = {d}', fontsize=13)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Per-round LER vs Measurement Rounds (Long Range)\n'
                 f'\u03b4={delta:.3f}, p_gate={p_gate:.2e}',
                 fontsize=13, y=1.03)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {save_path}")
    plt.show()


def plot_improvement_ratio(all_results, code_distances, p_gate, delta,
                            save_path="rounds_improvement_ratio.pdf"):
    """Plot pL_raw / pL_erasure vs T for each d."""
    n_d = len(code_distances)
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {3: 'C0', 5: 'C1', 7: 'C2', 9: 'C3', 11: 'C4', 13: 'C5', 15: 'C6'}

    for d in code_distances:
        if 'raw' not in all_results or d not in all_results['raw']:
            continue
        if 'erasure' not in all_results or d not in all_results['erasure']:
            continue

        raw = all_results['raw'][d]
        era = all_results['erasure'][d]

        # Find common T values
        raw_T_map = {T: i for i, T in enumerate(raw["T"])}
        common_T, ratio_vals = [], []
        for j, T in enumerate(era["T"]):
            if T in raw_T_map:
                pL_raw = raw["pL"][raw_T_map[T]]
                pL_era = era["pL"][j]
                if pL_raw > 0 and pL_era > 0:
                    common_T.append(T)
                    ratio_vals.append(pL_raw / pL_era)

        if common_T:
            clr = colors.get(d, 'gray')
            ax.plot(common_T, ratio_vals, 'o-', color=clr,
                    markersize=4, linewidth=1.2, label=f'd={d}')

    ax.axhline(y=1, color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel('Measurement rounds $T$', fontsize=12)
    ax.set_ylabel('$p_L^{\\mathrm{raw}} / p_L^{\\mathrm{erasure}}$', fontsize=14)
    ax.set_title(f'Improvement Ratio (Raw/Erasure) vs Rounds\n'
                 f'\u03b4={delta:.3f}, p_gate={p_gate:.2e}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {save_path}")
    plt.show()


# ============== Main ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Long-range Rounds Comparison: Raw vs Erasure')
    # Erasure conversion (Rm/Rc) from CSV
    parser.add_argument('--sweep-csv', required=True,
                        help='erasure_unsup_sweep.csv file path')
    parser.add_argument('--delta', type=float, default=0.475,
                        help='Delta value for erasure Rm/Rc (default: 0.475)')
    # Per-scenario Pm and atom loss
    parser.add_argument('--pm-raw', type=float, required=True,
                        help='Measurement error rate for Raw scenario')
    parser.add_argument('--pm-erasure', type=float, required=True,
                        help='Measurement error rate for Erasure scenario')
    parser.add_argument('--pl-raw', type=float, default=0.0,
                        help='Atom loss probability for Raw scenario (default: 0)')
    parser.add_argument('--pl-erasure', type=float, default=0.0,
                        help='Atom loss probability for Erasure scenario (default: 0)')
    # Simulation settings
    parser.add_argument('--mode', choices=['params', 'fast', 'full', 'plot'],
                        default='params',
                        help='Mode: params, fast, full, plot')
    parser.add_argument('--p-gate', type=float, default=0.0005,
                        help='Fixed gate error rate (default: 0.0005)')
    parser.add_argument('--code-distances', type=int, nargs='+', default=None,
                        help='Code distances (default: fast=[5,7,9], full=[5,7,9,11,13])')
    parser.add_argument('--max-rounds', type=int, default=100,
                        help='Maximum measurement rounds (default: 100)')
    parser.add_argument('--round-step', type=int, default=None,
                        help='Fixed step between rounds (default: adaptive)')
    parser.add_argument('--data-dir', default=None,
                        help='Directory for saving/loading result files')
    parser.add_argument('--output', default=None,
                        help='Output plot file path')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers (0 = all cores, 1 = sequential)')
    parser.add_argument('--refresh', action='store_true',
                        help='Discard previous results and start fresh')
    args = parser.parse_args()
    args.parallel = resolve_parallel_workers(args.parallel)

    # Parse sweep CSV for Rm/Rc
    sweep_rows = parse_sweep_csv(args.sweep_csv)
    row = find_row_by_delta(sweep_rows, args.delta)
    _, Rm, Rc, delta_actual, N_era = get_params_from_row(row)

    Pm_raw = args.pm_raw
    Pm_erasure = args.pm_erasure
    pl_raw = args.pl_raw
    pl_erasure = args.pl_erasure
    p_gate = args.p_gate
    max_rounds = args.max_rounds

    # Print parameters
    ratio = Rm / Rc if Rc > 0 else float('inf')
    print(f"\n{'='*70}")
    print(f"  Rounds Comparison (Long Range): Raw vs Erasure")
    print(f"  delta = {delta_actual:.3f}, p_gate = {p_gate:.2e}")
    print(f"{'='*70}")
    print(f"  [Raw]     Pm = {Pm_raw:.6f},  pl = {pl_raw:.6f}")
    print(f"  [Erasure] Pm = {Pm_erasure:.6f},  pl = {pl_erasure:.6f}")
    print(f"  Rm  = {Rm:.6f}  (P(erasure | error))")
    print(f"  Rc  = {Rc:.8f}  (P(erasure | correct))")
    print(f"  Rm/Rc ratio = {ratio:.1f}x")
    print(f"  Max rounds  = {max_rounds}")
    print(f"{'='*70}")

    if args.mode == 'params':
        sys.exit(0)

    # Setup
    delta_tag = f"d{delta_actual:.3f}".replace('.', 'p')
    data_dir = args.data_dir or f"results_rounds_long_{delta_tag}_p{p_gate:.0e}"
    os.makedirs(data_dir, exist_ok=True)
    results_file = os.path.join(data_dir, "results_rounds_long.json")
    output = args.output or os.path.join(data_dir, "rounds_comparison_long.pdf")

    if args.mode == 'plot':
        if not os.path.exists(results_file):
            print(f"Error: Results file not found: {results_file}")
            sys.exit(1)

        all_results, params = load_results(results_file)
        code_distances = sorted(set(
            d for sc in all_results for d in all_results[sc]
        ))

        plot_rounds_comparison(all_results, code_distances, p_gate, delta_actual,
                               Pm_raw, Pm_erasure, Rm, Rc, save_path=output)

        ratio_path = output.replace('.pdf', '_ratio.pdf')
        plot_improvement_ratio(all_results, code_distances, p_gate, delta_actual,
                                save_path=ratio_path)
        sys.exit(0)

    # Simulation modes
    compile_code_if_necessary()

    if args.mode == 'fast':
        code_distances = args.code_distances or [5, 7, 9]
        runtime_budget = (200, 30)
    else:  # full
        code_distances = args.code_distances or [5, 7, 9, 11, 13]
        runtime_budget = (40000, 3600)

    # Build per-d rounds list
    rounds_list = {}
    for d in code_distances:
        rounds_list[d] = build_rounds_list(d, max_rounds, step=args.round_step)

    total_rounds_per_d = {d: len(rounds_list[d]) for d in code_distances}
    print(f"\n  Code distances: {code_distances}")
    for d in code_distances:
        rlist = rounds_list[d]
        print(f"    d={d}: {len(rlist)} points, T={rlist[0]}..{rlist[-1]}")

    # Load existing results (checkpoint/resume)
    all_results = {}
    if not args.refresh and os.path.exists(results_file):
        all_results, _ = load_results(results_file)
        total_existing = sum(
            len(all_results[sc][d]["T"])
            for sc in all_results for d in all_results[sc]
        )
        print(f"\n  Loaded existing results: {total_existing} data points")
    elif args.refresh:
        print(f"\n  --refresh: starting fresh")

    # Determine remaining work per scenario
    n_scenarios = 2
    scenarios_config = [
        ('raw',     'Raw',     make_noise_config_raw(Pm_raw, pl_raw),         Pm_raw,     pl_raw),
        ('erasure', 'Erasure', make_noise_config_erasure(Pm_erasure, Rm, Rc, pl_erasure), Pm_erasure, pl_erasure),
    ]

    for sc_idx, (sc_key, sc_label, noise_config, sc_pm, sc_pl) in enumerate(scenarios_config):
        existing_sc = all_results.get(sc_key, {})
        remaining, skipped = filter_remaining_rounds(rounds_list, existing_sc)

        if not remaining:
            print(f"\n  [{sc_idx+1}/{n_scenarios}] {sc_label}: all done (skipped {skipped})")
            continue

        total_remaining = sum(len(v) for v in remaining.values())
        total_total = sum(len(v) for v in rounds_list.values())
        print(f"\n{'='*60}")
        print(f" [{sc_idx+1}/{n_scenarios}] {sc_label} (Pm={sc_pm:.6f}, pl={sc_pl:.6f})")
        if skipped > 0:
            print(f"   Resuming: {skipped}/{total_total} already done, {total_remaining} remaining")
        print(f"{'='*60}")

        # Only sweep remaining (d, T) pairs
        remaining_distances = sorted(remaining.keys())
        tracker = ProgressTracker(total_remaining, "simulations",
                                  print_every=max(1, len(remaining_distances)))

        if args.parallel > 1:
            new_results = run_rounds_sweep_parallel(
                sc_label, p_gate, remaining_distances, remaining,
                noise_config, runtime_budget, args.parallel, tracker=tracker
            )
        else:
            new_results = run_rounds_sweep_sequential(
                sc_label, p_gate, remaining_distances, remaining,
                noise_config, runtime_budget, tracker=tracker
            )

        tracker.summary()

        # Merge into all_results
        if sc_key not in all_results:
            all_results[sc_key] = {}
        all_results[sc_key] = merge_checkpoint(all_results[sc_key], new_results)

        # Save checkpoint after each scenario
        save_results(all_results, {
            "delta": delta_actual, "p_gate": p_gate,
            "Pm_raw": Pm_raw, "Pm_erasure": Pm_erasure,
            "pl_raw": pl_raw, "pl_erasure": pl_erasure,
            "Rm": Rm, "Rc": Rc,
            "code_distances": code_distances,
            "max_rounds": max_rounds,
        }, results_file)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  Summary: Per-round LER at selected T values")
    print(f"{'='*80}")
    for d in code_distances:
        print(f"\n  d = {d}:")
        header = f"  {'T':>4s}  {'pL_Raw':>12s}  {'pL_Erasure':>12s}  {'ratio':>8s}"
        print(header)
        print(f"  {'-'*(len(header)-2)}")

        T_set = set()
        for sc in ['raw', 'erasure']:
            if sc in all_results and d in all_results[sc]:
                T_set.update(all_results[sc][d]["T"])
        # Show subset of T values
        T_all = sorted(T_set)
        show_T = [T for T in T_all if T <= d + 2 or T % 10 == 0 or T == T_all[-1]]

        for T in show_T:
            row_str = f"  {T:>4d}"
            pL_vals = {}
            for sc in ['raw', 'erasure']:
                if sc in all_results and d in all_results[sc]:
                    data = all_results[sc][d]
                    if T in data["T"]:
                        idx = data["T"].index(T)
                        pL_vals[sc] = data["pL"][idx]
                        row_str += f"  {data['pL'][idx]:>12.4e}"
                    else:
                        row_str += f"  {'---':>12s}"
                else:
                    row_str += f"  {'N/A':>12s}"
            if 'raw' in pL_vals and 'erasure' in pL_vals and pL_vals['erasure'] > 0:
                row_str += f"  {pL_vals['raw']/pL_vals['erasure']:>8.2f}x"
            print(row_str)

    # Plot
    plot_rounds_comparison(all_results, code_distances, p_gate, delta_actual,
                           Pm_raw, Pm_erasure, Rm, Rc, save_path=output)

    ratio_path = output.replace('.pdf', '_ratio.pdf')
    plot_improvement_ratio(all_results, code_distances, p_gate, delta_actual,
                            save_path=ratio_path)
