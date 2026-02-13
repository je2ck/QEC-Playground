#!/usr/bin/env python3
"""
Measurement Rounds Comparison: Raw vs Den vs Den+Erasure

고정된 gate error p와 code distance d에서 measurement rounds T를 바꿔가며
LER(logical error rate)을 비교합니다.

목표: Den+Erasure가 같은 LER을 더 적은 rounds로 달성할 수 있는지 확인.

Three scenarios:
  1. Raw:          Pm_raw  (Raw row),  no erasure
  2. Den:          Pm_den  (Den row),  no erasure
  3. Den+Erasure:  Pm_den + soft erasure classes (from amb zone CSV)

X-axis: measurement rounds T
Y-axis: logical error rate pL

Usage:
    # 파라미터 확인
    python rounds_comparison.py --csv confusion_amb.csv --exposure 8 --mode params

    # quick 시뮬레이션 (고정 p=0.001, d=5,7,9)
    python rounds_comparison.py --csv confusion_amb.csv --exposure 8 --mode quick --p-gate 0.001

    # full 시뮬레이션
    python rounds_comparison.py --csv confusion_amb.csv --exposure 8 --mode full --p-gate 0.001

    # 플롯만
    python rounds_comparison.py --csv confusion_amb.csv --exposure 8 --mode plot --p-gate 0.001

    # rounds 범위 지정
    python rounds_comparison.py --csv confusion_amb.csv --exposure 8 --mode quick --p-gate 0.001 --rounds 1 3 5 7 9 11 13 15
"""

import os
import sys
import json
import csv
import argparse
import subprocess
from collections import defaultdict

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
from utils import ProgressTracker, run_parallel_simulations, scaled_runtime_budget, resolve_parallel_workers


# ============== CSV Parsing (reuse from confusion_amb_threshold.py) ==============

def parse_confusion_csv(csv_path):
    """Parse confusion matrix CSV → {exposure: {method: {TP, TN, FP, FN}}}"""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            exposure = int(row['exposure'].strip())
            method = row['Method'].strip()
            if exposure not in data:
                data[exposure] = {}
            data[exposure][method] = {
                'TP': int(row['TP'].strip()),
                'TN': int(row['TN'].strip()),
                'FP': int(row['FP'].strip()),
                'FN': int(row['FN'].strip()),
            }
    return data


def parse_amb_zone_csv(csv_path):
    """Parse per-sample ambiguous zone CSV → per-class counts."""
    counts = defaultdict(lambda: defaultdict(int))
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_cls = row['WeightClass'].strip()
            if raw_cls.endswith('L') or raw_cls.endswith('R'):
                cls = raw_cls[:-1]
            else:
                cls = raw_cls
            cat = row['Category'].strip()
            counts[cls][cat] += 1
    return counts


def compute_Pm(method_data):
    """Compute measurement error rate from TP/TN/FP/FN."""
    total_wrong = method_data['FP'] + method_data['FN']
    total = method_data['TP'] + method_data['TN'] + total_wrong
    return total_wrong / total if total > 0 else 0


def compute_erasure_classes(confusion_data, amb_counts, exposure):
    """
    Compute per-class Rm, Rc, and Bayes weights for Den+Erasure scenario.

    Returns:
        (Pm_den, classes_list)
    """
    den = confusion_data[exposure]['Den']
    total_wrong = den['FP'] + den['FN']
    total_correct = den['TP'] + den['TN']
    total = total_wrong + total_correct
    Pm_den = total_wrong / total if total > 0 else 0

    classes = []
    for cls_name in sorted(amb_counts, key=lambda x: int(x[1:])):
        fp = amb_counts[cls_name].get('FP', 0)
        fn = amb_counts[cls_name].get('FN', 0)
        tp = amb_counts[cls_name].get('TP', 0)
        tn = amb_counts[cls_name].get('TN', 0)

        wrong_k = fp + fn
        correct_k = tp + tn

        Rm_k = wrong_k / total_wrong if total_wrong > 0 else 0
        Rc_k = correct_k / total_correct if total_correct > 0 else 0

        num = Pm_den * Rm_k
        den_val = Pm_den * Rm_k + (1 - Pm_den) * Rc_k
        weight_k = num / den_val if den_val > 0 else 0

        classes.append({
            'name': cls_name,
            'Rm': Rm_k,
            'Rc': Rc_k,
            'weight': weight_k,
        })

    return Pm_den, classes


# ============== Simulate Functions ==============

def run_single_simulation(p, d, T, noise_config, decoder_config, runtime_budget):
    """
    Run a single QEC simulation with given parameters.

    Args:
        p: gate error rate
        d: code distance
        T: number of noisy measurement rounds
        noise_config: noise model JSON config dict
        decoder_config: decoder JSON config dict
        runtime_budget: (min_error_cases, time_budget)

    Returns:
        (pL, pL_dev)
    """
    min_error_cases, time_budget = runtime_budget

    parameters = [
        "--code-type", "rotated-planar-code",
        "--noise-model-builder", "only-gate-error-circuit-level",
        "--noise-model-configuration", json.dumps(noise_config),
        "--decoder", "union-find",
        "--decoder-config", json.dumps(decoder_config),
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


def make_noise_config_raw(Pm_raw):
    """Raw: measurement_error_rate = Pm_raw, no erasure."""
    return {
        "use_correlated_pauli": True,
        "use_correlated_erasure": True,
        "measurement_error_rate": Pm_raw,
    }


def make_noise_config_den(Pm_den):
    """Den: measurement_error_rate = Pm_den, no erasure."""
    return {
        "use_correlated_pauli": True,
        "use_correlated_erasure": True,
        "measurement_error_rate": Pm_den,
    }


def make_noise_config_den_erasure(Pm_den, classes):
    """Den + Erasure: measurement_error_rate_total + erasure_classes."""
    return {
        "use_correlated_pauli": True,
        "use_correlated_erasure": True,
        "measurement_error_rate_total": Pm_den,
        "erasure_classes": [
            {"Rm": round(c['Rm'], 8), "Rc": round(c['Rc'], 8)}
            for c in classes
        ],
    }


# ============== Rounds Sweep ==============

def run_rounds_sweep(label, p_gate, d, rounds_list, noise_config, decoder_config, runtime_budget, tracker=None):
    """
    Sweep measurement rounds T for a fixed (p, d).

    Returns:
        {"T": [...], "pL": [...], "pL_dev": [...]}
    """
    result = {"T": [], "pL": [], "pL_dev": []}

    for T in rounds_list:
        if tracker:
            tracker.begin_task()
        pL, pL_dev = run_single_simulation(
            p_gate, d, T, noise_config, decoder_config, runtime_budget
        )
        print(f"  [{label}] d={d:2d}, T={T:2d}: pL={pL:.4e} ± {pL_dev:.2e}")
        result["T"].append(T)
        result["pL"].append(pL)
        result["pL_dev"].append(pL_dev)
        if tracker:
            tracker.end_task()

    return result


def run_rounds_sweep_parallel(label, p_gate, code_distances, rounds_list,
                              noise_config, decoder_config, runtime_budget,
                              n_workers, tracker=None):
    """
    Parallel sweep: all (d, T) pairs for one scenario submitted concurrently.

    Returns:
        {d: {"T": [...], "pL": [...], "pL_dev": [...]}}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tasks = [(d, T) for d in code_distances for T in rounds_list]
    print(f"  \u26a1 [{label}] Parallel: {n_workers} workers, {len(tasks)} tasks")

    d_base = min(code_distances)
    result_map = {}
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_key = {}
        for d, T in tasks:
            budget = scaled_runtime_budget(runtime_budget, d, d_base)
            future = executor.submit(
                run_single_simulation, p_gate, d, T,
                noise_config, decoder_config, budget
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
            print(f"  [{label}] d={d:2d}, T={T:2d}: pL={pL:.4e} \u00b1 {pL_dev:.2e}")
            if tracker:
                tracker.task_done()

    # Organize by code distance, preserving T order
    results = {}
    for d in code_distances:
        results[d] = {"T": [], "pL": [], "pL_dev": []}
        for T in rounds_list:
            pL, pL_dev = result_map[(d, T)]
            results[d]["T"].append(T)
            results[d]["pL"].append(pL)
            results[d]["pL_dev"].append(pL_dev)

    return results


# ============== I/O ==============

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


# ============== Plot ==============

def plot_rounds_comparison(all_results, code_distances, p_gate, exposure,
                           Pm_raw, Pm_den, save_path="rounds_comparison.pdf"):
    """
    Plot LER vs measurement rounds T for Raw / Den / Den+Erasure.

    One subplot per code distance.
    """
    n_d = len(code_distances)
    fig, axes = plt.subplots(1, n_d, figsize=(5.5 * n_d, 5), squeeze=False)

    scenario_styles = {
        'raw':          {'label': f'Raw (Pm={Pm_raw:.4f})',         'color': 'C3', 'marker': '^', 'ls': '--'},
        'den':          {'label': f'Den (Pm={Pm_den:.4f})',         'color': 'C1', 'marker': 's', 'ls': '--'},
        'den_erasure':  {'label': f'Den+Erasure (Pm={Pm_den:.4f})', 'color': 'C0', 'marker': 'o', 'ls': '-'},
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

            # Per-round LER (probability): 1 - (1 - pL_shot)^(1/T)
            pL_per_round = 1 - (1 - pL_arr[valid]) ** (1.0 / T_arr[valid])

            ax.plot(T_arr[valid], pL_per_round,
                    marker=style['marker'], linestyle=style['ls'],
                    color=style['color'], markersize=6, linewidth=1.5,
                    label=style['label'])

        # d rounds reference line
        ax.axvline(x=d, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(d + 0.3, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-5,
                f'T=d={d}', fontsize=8, color='gray', rotation=90, va='bottom')

        ax.set_yscale('log')
        ax.set_xlabel('Measurement rounds $T$', fontsize=12)
        ax.set_ylabel('Per-round logical error rate', fontsize=12)
        ax.set_title(f'd = {d}', fontsize=13)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # Integer x ticks
        T_all = []
        for sc in all_results:
            if d in all_results[sc]:
                T_all.extend(all_results[sc][d]["T"])
        if T_all:
            ax.set_xticks(sorted(set(T_all)))

    fig.suptitle(f'Per-round LER vs Measurement Rounds\n'
                 f'Exposure={exposure}ms, p_gate={p_gate:.2e}',
                 fontsize=13, y=1.03)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {save_path}")
    plt.show()


def plot_round_savings(all_results, code_distances, p_gate, exposure,
                       Pm_raw, Pm_den, save_path="round_savings.pdf"):
    """
    Plot "round savings": for each target LER, how many rounds does each method need?

    Uses interpolation to find T_min for target LER levels.
    One subplot per code distance.
    """
    n_d = len(code_distances)
    fig, axes = plt.subplots(1, n_d, figsize=(5.5 * n_d, 5), squeeze=False)

    scenario_styles = {
        'raw':          {'label': 'Raw',         'color': 'C3', 'marker': '^'},
        'den':          {'label': 'Den',         'color': 'C1', 'marker': 's'},
        'den_erasure':  {'label': 'Den+Erasure', 'color': 'C0', 'marker': 'o'},
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
            if np.sum(valid) < 2:
                continue

            # Sort by T
            order = np.argsort(T_arr[valid])
            T_sorted = T_arr[valid][order]
            pL_sorted = pL_arr[valid][order]

            # Per-round LER (probability): 1 - (1 - pL_shot)^(1/T)
            pL_per_round = 1 - (1 - pL_sorted) ** (1.0 / T_sorted)

            ax.plot(T_sorted, pL_per_round,
                    marker=style['marker'], linestyle='-',
                    color=style['color'], markersize=6, linewidth=1.5,
                    label=style['label'])

        ax.set_yscale('log')
        ax.set_xlabel('Measurement rounds $T$', fontsize=12)
        ax.set_ylabel('Per-round logical error rate', fontsize=12)
        ax.set_title(f'd = {d}', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Per-round LER vs Measurement Rounds\n'
                 f'Exposure={exposure}ms, p_gate={p_gate:.2e}',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {save_path}")
    plt.show()


def print_round_savings_table(all_results, code_distances):
    """
    Print a table showing for each method & d, the LER at each T.
    Highlight where Den+Erasure matches Raw's LER with fewer rounds.
    """
    print(f"\n{'='*80}")
    print(f"  Round Savings Summary")
    print(f"{'='*80}")

    for d in code_distances:
        print(f"\n  d = {d}:")
        # Collect all T values
        T_set = set()
        for sc in ['raw', 'den', 'den_erasure']:
            if sc in all_results and d in all_results[sc]:
                T_set.update(all_results[sc][d]["T"])
        T_list = sorted(T_set)

        header = f"  {'T':>4s}"
        for sc_name in ['Raw', 'Den', 'Den+Era']:
            header += f"  {'pL_'+sc_name:>14s}"
        print(header)
        print(f"  {'-'*(len(header)-2)}")

        for T in T_list:
            row = f"  {T:>4d}"
            for sc in ['raw', 'den', 'den_erasure']:
                if sc in all_results and d in all_results[sc]:
                    data = all_results[sc][d]
                    if T in data["T"]:
                        idx = data["T"].index(T)
                        pL = data["pL"][idx]
                        row += f"  {pL:>14.4e}"
                    else:
                        row += f"  {'---':>14s}"
                else:
                    row += f"  {'N/A':>14s}"
            print(row)

        # Find round savings: at which T does Den+Erasure reach Raw's T=d LER?
        raw_key = 'raw'
        era_key = 'den_erasure'
        if raw_key in all_results and d in all_results[raw_key] and \
           era_key in all_results and d in all_results[era_key]:
            raw_data = all_results[raw_key][d]
            era_data = all_results[era_key][d]
            # Raw LER at T=d
            if d in raw_data["T"]:
                raw_idx = raw_data["T"].index(d)
                raw_pL_at_d = raw_data["pL"][raw_idx]
                # Find minimum T where Den+Erasure achieves ≤ raw_pL_at_d
                for j, T in enumerate(era_data["T"]):
                    if era_data["pL"][j] <= raw_pL_at_d and era_data["pL"][j] > 0:
                        savings = d - T
                        pct = savings / d * 100
                        print(f"\n  ★ Den+Erasure reaches Raw(T={d}) LER={raw_pL_at_d:.2e}"
                              f" at T={T} → saves {savings} rounds ({pct:.0f}%)")
                        break


# ============== Main ==============

if __name__ == "__main__":
    default_amb_dir = os.path.join(
        os.path.expanduser("~"),
        "Documents/research/denoise/Noise2NoiseFlow/noise2noiseflow/uncertainty_weighted_outputs"
    )

    parser = argparse.ArgumentParser(
        description='Measurement Rounds Comparison: Raw vs Den vs Den+Erasure')
    parser.add_argument('--csv', required=True,
                        help='Confusion matrix CSV file (with Raw/Den/Amb rows)')
    parser.add_argument('--amb-dir', default=default_amb_dir,
                        help='Directory containing ambiguous_zone_data_Xms.csv files')
    parser.add_argument('--exposure', type=int, required=True,
                        help='Exposure time to use')
    parser.add_argument('--mode', choices=['params', 'quick', 'full', 'plot'],
                        default='params',
                        help='Mode: params, quick, full, plot')
    parser.add_argument('--p-gate', type=float, default=0.0005,
                        help='Fixed gate error rate (default: 0.0005)')
    parser.add_argument('--code-distances', type=int, nargs='+', default=None,
                        help='Code distances to simulate (default: quick=[5,7,9], full=[5,7,9,11])')
    parser.add_argument('--rounds', type=int, nargs='+', default=None,
                        help='Specific rounds to test (e.g. --rounds 1 3 5 7 9 11 13 15)')
    parser.add_argument('--max-half-weight', type=int, default=23,
                        help='UF decoder max_half_weight (default: 23)')
    parser.add_argument('--data-dir', default=None,
                        help='Directory for saving/loading result files')
    parser.add_argument('--output', default=None,
                        help='Output plot file path')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers (0 = all cores, 1 = sequential)')
    args = parser.parse_args()
    args.parallel = resolve_parallel_workers(args.parallel)

    # Parse confusion CSV
    confusion_data = parse_confusion_csv(args.csv)

    exp = args.exposure
    if exp not in confusion_data:
        print(f"Error: Exposure {exp} not found. Available: {sorted(confusion_data.keys())}")
        sys.exit(1)

    # Get Pm for Raw and Den
    if 'Raw' not in confusion_data[exp]:
        print(f"Error: No Raw row for exposure {exp}")
        sys.exit(1)
    if 'Den' not in confusion_data[exp]:
        print(f"Error: No Den row for exposure {exp}")
        sys.exit(1)

    Pm_raw = compute_Pm(confusion_data[exp]['Raw'])
    Pm_den = compute_Pm(confusion_data[exp]['Den'])

    # Parse amb zone CSV for erasure classes
    amb_csv = os.path.join(args.amb_dir, f"ambiguous_zone_data_{exp}ms.csv")
    if not os.path.exists(amb_csv):
        print(f"Warning: Amb zone CSV not found: {amb_csv}")
        print(f"         Den+Erasure scenario will be skipped.")
        Pm_den_era, classes = Pm_den, []
    else:
        amb_counts = parse_amb_zone_csv(amb_csv)
        Pm_den_era, classes = compute_erasure_classes(confusion_data, amb_counts, exp)

    p_gate = args.p_gate
    mhw = args.max_half_weight

    # ============== Print parameters ==============
    print(f"\n{'='*70}")
    print(f"  Measurement Rounds Comparison")
    print(f"  Exposure = {exp}ms, p_gate = {p_gate:.2e}")
    print(f"{'='*70}")
    print(f"  Pm_raw = {Pm_raw:.6f}  (Raw measurement error rate)")
    print(f"  Pm_den = {Pm_den:.6f}  (Den measurement error rate)")
    print(f"  Pm_raw / Pm_den = {Pm_raw/Pm_den:.2f}x improvement from denoising")
    if classes:
        max_wt = max(c['weight'] for c in classes) if classes else 1.0
        print(f"  Erasure classes: {len(classes)}")
        for c in classes:
            nw = c['weight'] / max_wt if max_wt > 0 else 0
            print(f"    {c['name']}: Rm={c['Rm']:.4f}, Rc={c['Rc']:.4f}, NormWt={nw:.4f}")
    print(f"  max_half_weight = {mhw}")
    print(f"{'='*70}")

    if args.mode == 'params':
        sys.exit(0)

    # Setup
    data_dir = args.data_dir or f"results_rounds_exp{exp}_p{p_gate:.0e}"
    os.makedirs(data_dir, exist_ok=True)
    output = args.output or os.path.join(data_dir, f"rounds_comparison_exp{exp}.pdf")

    compile_code_if_necessary()

    decoder_config = {"pcmg": True, "max_half_weight": mhw}

    if args.mode in ('quick', 'full'):
        # Determine rounds to sweep and code distances
        if args.mode == 'quick':
            code_distances = args.code_distances or [5, 7, 9]
            runtime_budget = (200, 30)
        else:  # full
            code_distances = args.code_distances or [5, 7, 9, 11, 13]
            runtime_budget = (40000, 3600)

        if args.rounds:
            rounds_list = sorted(args.rounds)
        else:
            # Default: 1 to 2*max(d), odd numbers for symmetry
            max_d = max(code_distances)
            rounds_list = list(range(1, 2 * max_d + 2, 2))

        print(f"\n  Code distances: {code_distances}")
        print(f"  Rounds to test: {rounds_list}")

        # ============== Run simulations ==============
        all_results = {}
        d_base = min(code_distances)

        n_scenarios = 3 if classes else 2
        total_sims = n_scenarios * len(code_distances) * len(rounds_list)
        tracker = ProgressTracker(total_sims, "simulations", print_every=len(rounds_list))

        # --- 1/3: Raw ---
        print(f"\n{'='*60}")
        print(f" [1/3] Raw (Pm={Pm_raw:.6f})")
        print(f"{'='*60}")
        noise_raw = make_noise_config_raw(Pm_raw)
        if args.parallel > 1:
            all_results['raw'] = run_rounds_sweep_parallel(
                'raw', p_gate, code_distances, rounds_list, noise_raw,
                decoder_config, runtime_budget, args.parallel, tracker=tracker
            )
        else:
            all_results['raw'] = {}
            for d in code_distances:
                print(f"\n  --- d = {d} ---")
                all_results['raw'][d] = run_rounds_sweep(
                    'raw', p_gate, d, rounds_list, noise_raw, decoder_config,
                    scaled_runtime_budget(runtime_budget, d, d_base),
                    tracker=tracker
                )

        # --- 2/3: Den ---
        print(f"\n{'='*60}")
        print(f" [2/3] Den (Pm={Pm_den:.6f})")
        print(f"{'='*60}")
        noise_den = make_noise_config_den(Pm_den)
        if args.parallel > 1:
            all_results['den'] = run_rounds_sweep_parallel(
                'den', p_gate, code_distances, rounds_list, noise_den,
                decoder_config, runtime_budget, args.parallel, tracker=tracker
            )
        else:
            all_results['den'] = {}
            for d in code_distances:
                print(f"\n  --- d = {d} ---")
                all_results['den'][d] = run_rounds_sweep(
                    'den', p_gate, d, rounds_list, noise_den, decoder_config,
                    scaled_runtime_budget(runtime_budget, d, d_base),
                    tracker=tracker
                )

        # --- 3/3: Den + Erasure ---
        if classes:
            print(f"\n{'='*60}")
            print(f" [3/3] Den + Erasure (Pm={Pm_den:.6f}, {len(classes)} classes)")
            print(f"{'='*60}")
            noise_era = make_noise_config_den_erasure(Pm_den, classes)
            if args.parallel > 1:
                all_results['den_erasure'] = run_rounds_sweep_parallel(
                    'den+era', p_gate, code_distances, rounds_list, noise_era,
                    decoder_config, runtime_budget, args.parallel, tracker=tracker
                )
            else:
                all_results['den_erasure'] = {}
                for d in code_distances:
                    print(f"\n  --- d = {d} ---")
                    all_results['den_erasure'][d] = run_rounds_sweep(
                        'den+era', p_gate, d, rounds_list, noise_era, decoder_config,
                        scaled_runtime_budget(runtime_budget, d, d_base),
                        tracker=tracker
                    )
        else:
            print("\n  [SKIP] Den+Erasure: no ambiguous zone data")

        tracker.summary()

        # ============== Save ==============
        save_results(all_results, {
            "exposure": exp, "p_gate": p_gate,
            "Pm_raw": Pm_raw, "Pm_den": Pm_den,
            "code_distances": code_distances,
            "rounds_list": rounds_list,
            "max_half_weight": mhw,
        }, os.path.join(data_dir, "results_rounds.json"))

        # ============== Print summary & Plot ==============
        print_round_savings_table(all_results, code_distances)

        plot_rounds_comparison(all_results, code_distances, p_gate, exp,
                               Pm_raw, Pm_den, save_path=output)

        savings_path = output.replace('.pdf', '_savings.pdf')
        plot_round_savings(all_results, code_distances, p_gate, exp,
                           Pm_raw, Pm_den, save_path=savings_path)

    elif args.mode == 'plot':
        results_file = os.path.join(data_dir, "results_rounds.json")
        if not os.path.exists(results_file):
            print(f"Error: Results file not found: {results_file}")
            sys.exit(1)

        all_results, params = load_results(results_file)
        code_distances = sorted(set(
            d for sc in all_results for d in all_results[sc]
        ))

        print_round_savings_table(all_results, code_distances)

        plot_rounds_comparison(all_results, code_distances, p_gate, exp,
                               Pm_raw, Pm_den, save_path=output)

        savings_path = output.replace('.pdf', '_savings.pdf')
        plot_round_savings(all_results, code_distances, p_gate, exp,
                           Pm_raw, Pm_den, save_path=savings_path)
