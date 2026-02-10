#!/usr/bin/env python3
"""
Ambiguous Zone Erasure Threshold Analysis (with Soft Weights)

Key difference from confusion_threshold.py:
  - ALL measurements in the ambiguous zone are treated as erasures
    (confusion_threshold.py used only E-marked measurements)
  - Supports multi-class Bayes weights from CSV WeightClass labels
  - Handles both C1L/C1R (merged) and C1 formats
  - Number of classes is determined by the CSV (not hardcoded)
  - Weights are normalized so that the highest-confidence class = 1.0

Two simulation scenarios compared:
  1. soft:       Multi-class weighted erasure (per-class Bayes weight, normalized)
  2. no-erasure: No erasure (pure measurement error baseline)

Parameters derived from:
  - confusion_amb.csv: Den row → Pm (total measurement error rate)
  - ambiguous_zone_data_Xms.csv: per-sample WeightClass → Rm_k, Rc_k per class

Math:
  Rm_k = (FP_k + FN_k in amb) / (FP + FN in Den)     = P(class k erasure | error)
  Rc_k = (TP_k + TN_k in amb) / (TP + TN in Den)     = P(class k erasure | correct)
  Bayes weight: w_k = Pm·Rm_k / (Pm·Rm_k + (1-Pm)·Rc_k)
  Normalized:   w_k' = w_k / max(w_k)  → highest class gets weight 1.0

Usage:
    python confusion_amb_threshold.py --csv confusion_amb.csv --exposure 8 --mode params
    python confusion_amb_threshold.py --csv confusion_amb.csv --exposure 8 --mode quick
    python confusion_amb_threshold.py --csv confusion_amb.csv --exposure 8 --mode full
    python confusion_amb_threshold.py --csv confusion_amb.csv --exposure 8 --mode plot
    python confusion_amb_threshold.py --csv confusion_amb.csv --list-exposures
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
from utils import (
    find_crossing_point, estimate_threshold_from_data,
    compute_lambda_factor, print_lambda_summary, plot_lambda_comparison,
)


# ============== CSV Parsing ==============

def parse_confusion_csv(csv_path):
    """
    Parse confusion matrix CSV (confusion_amb.csv).

    Returns:
        {exposure: {method: {"TP": int, "TN": int, "FP": int, "FN": int}}}
    """
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
    """
    Parse per-sample ambiguous zone CSV → per-class counts.

    Handles both formats:
      - C1L, C1R, C2L, ... → merged to C1, C2, ...
      - C1, C2, ...        → used as-is

    Returns:
        {class_name: {"TP": int, "TN": int, "FP": int, "FN": int}}
    """
    counts = defaultdict(lambda: defaultdict(int))
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_cls = row['WeightClass'].strip()
            # Strip trailing L/R if present (C1L → C1, C4R → C4)
            if raw_cls.endswith('L') or raw_cls.endswith('R'):
                cls = raw_cls[:-1]
            else:
                cls = raw_cls
            cat = row['Category'].strip()
            counts[cls][cat] += 1
    return counts


def calculate_amb_params(confusion_data, amb_counts, exposure):
    """
    Calculate Pm, per-class Rm_k, Rc_k, and Bayes weights.

    Uses Den row from confusion CSV as denominator (total errors/correct counts).
    Uses per-class ambiguous zone counts as numerators.

    Returns:
        (Pm, classes_list)
        where classes_list = [{'name', 'Rm', 'Rc', 'weight', 'wrong', 'correct'}, ...]
    """
    den = confusion_data[exposure]['Den']
    total_wrong = den['FP'] + den['FN']
    total_correct = den['TP'] + den['TN']
    total = total_wrong + total_correct
    Pm = total_wrong / total if total > 0 else 0

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

        # Bayes weight: P(error | class k erasure)
        num = Pm * Rm_k
        den_val = Pm * Rm_k + (1 - Pm) * Rc_k
        weight_k = num / den_val if den_val > 0 else 0

        classes.append({
            'name': cls_name,
            'Rm': Rm_k,
            'Rc': Rc_k,
            'weight': weight_k,
            'wrong': wrong_k,
            'correct': correct_k,
        })

    return Pm, classes


def print_params_summary(Pm, classes, exposure, confusion_data):
    """Print parameter summary."""
    den = confusion_data[exposure]['Den']
    total_wrong = den['FP'] + den['FN']
    total_correct = den['TP'] + den['TN']
    total = total_wrong + total_correct

    sum_Rm = sum(c['Rm'] for c in classes)
    sum_Rc = sum(c['Rc'] for c in classes)
    total_amb_wrong = sum(c['wrong'] for c in classes)
    total_amb_correct = sum(c['correct'] for c in classes)

    print(f"\n{'='*70}")
    print(f"  Exposure = {exposure}ms  (All ambiguous zone → erasure)")
    print(f"{'='*70}")
    print(f"  Den: TP={den['TP']}, TN={den['TN']}, FP={den['FP']}, FN={den['FN']}  (total={total})")
    print(f"  Amb zone: {total_amb_wrong} wrong + {total_amb_correct} correct = {total_amb_wrong + total_amb_correct} samples")
    print()
    print(f"  Pm  = {total_wrong}/{total} = {Pm:.6f}")
    print(f"  Sum(Rm_k) = {sum_Rm:.4f}  ({total_amb_wrong}/{total_wrong} errors detected as amb)")
    print(f"  Sum(Rc_k) = {sum_Rc:.4f}  ({total_amb_correct}/{total_correct} correct flagged as amb)")
    print()

    max_wt = max(c['weight'] for c in classes) if classes else 1.0
    hdr = f"  {'Class':<6} {'Wrong':>6} {'Correct':>8} {'Rm_k':>8} {'Rc_k':>8} {'BayesWt':>8} {'NormWt':>8}"
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")
    for c in classes:
        norm_wt = c['weight'] / max_wt if max_wt > 0 else 0
        print(f"  {c['name']:<6} {c['wrong']:>6} {c['correct']:>8} "
              f"{c['Rm']:>8.4f} {c['Rc']:>8.4f} {c['weight']:>8.4f} {norm_wt:>8.4f}")

    # Derived noise model parameters
    hidden_rate = Pm * (1 - sum_Rm)
    total_erasure = Pm * sum_Rm + (1 - Pm) * sum_Rc

    print(f"\n  Derived noise model parameters:")
    print(f"    Hidden meas error:   Pm*(1-ΣRm) = {hidden_rate:.6f}")
    print(f"    Total erasure rate:  Pm*ΣRm + (1-Pm)*ΣRc = {total_erasure:.6f}")
    if total_erasure > 0:
        p_err_given_erasure = (Pm * sum_Rm) / total_erasure
        print(f"    P(error|erasure):    = {p_err_given_erasure:.6f}")
    print(f"{'='*70}")


# ============== Simulate Functions ==============

def create_simulate_func_soft(Pm, classes, max_half_weight=1):
    """
    Multi-class weighted erasure simulation.

    Uses measurement_error_rate_total + erasure_classes config.
    Each class gets a Bayes weight (normalized so max=1.0) in the decoder.
    """
    config = {
        "use_correlated_pauli": True,
        "use_correlated_erasure": True,
        "measurement_error_rate_total": Pm,
        "erasure_classes": [
            {"Rm": round(c['Rm'], 8), "Rc": round(c['Rc'], 8)}
            for c in classes
        ],
    }
    decoder_config = {"pcmg": True, "max_half_weight": max_half_weight}

    def simulate_func(p, d, runtime_budget, p_graph=None):
        min_error_cases, time_budget = runtime_budget
        noisy_measurements = d

        parameters = [
            "--code-type", "rotated-planar-code",
            "--noise-model-builder", "only-gate-error-circuit-level",
            "--noise-model-configuration", json.dumps(config),
            "--decoder", "union-find",
            "--decoder-config", json.dumps(decoder_config),
        ]

        command = qecp_benchmark_simulate_func_command_vec(
            p, d, d, noisy_measurements, parameters,
            min_error_cases=min_error_cases,
            time_budget=time_budget,
            p_graph=p_graph,
        )

        stdout, returncode = run_qecp_command_get_stdout(command)
        if returncode != 0:
            print(f"  [ERROR] soft simulation failed for p={p}, d={d}")
            return (0.5, 1.0)

        full_result = stdout.strip(" \r\n").split("\n")[-1]
        lst = full_result.split(" ")
        pL = float(lst[5])
        pL_dev = float(lst[7])

        print(f"  [soft] d={d:2d}, p={p:.4e}: pL={pL:.4e} ± {pL_dev:.2e}")
        return (pL, pL_dev)

    return simulate_func


def create_simulate_func_no_erasure(Pm, max_half_weight=1):
    """
    No erasure baseline: pure measurement error.

    Same physical Pm but no erasure detection.
    """
    config = {
        "use_correlated_pauli": True,
        "use_correlated_erasure": True,
        "measurement_error_rate": Pm,
    }
    decoder_config = {"pcmg": True, "max_half_weight": max_half_weight}

    def simulate_func(p, d, runtime_budget, p_graph=None):
        min_error_cases, time_budget = runtime_budget
        noisy_measurements = d

        parameters = [
            "--code-type", "rotated-planar-code",
            "--noise-model-builder", "only-gate-error-circuit-level",
            "--noise-model-configuration", json.dumps(config),
            "--decoder", "union-find",
            "--decoder-config", json.dumps(decoder_config),
        ]

        command = qecp_benchmark_simulate_func_command_vec(
            p, d, d, noisy_measurements, parameters,
            min_error_cases=min_error_cases,
            time_budget=time_budget,
            p_graph=p_graph,
        )

        stdout, returncode = run_qecp_command_get_stdout(command)
        if returncode != 0:
            print(f"  [ERROR] no-erasure simulation failed for p={p}, d={d}")
            return (0.5, 1.0)

        full_result = stdout.strip(" \r\n").split("\n")[-1]
        lst = full_result.split(" ")
        pL = float(lst[5])
        pL_dev = float(lst[7])

        print(f"  [no-erasure] d={d:2d}, p={p:.4e}: pL={pL:.4e} ± {pL_dev:.2e}")
        return (pL, pL_dev)

    return simulate_func


# ============== Sweep & I/O ==============

def run_p_sweep(simulate_func, code_distances, p_list, runtime_budget):
    """Run simulation over a sweep of physical error probabilities."""
    results = {d: {"p": [], "pL": [], "pL_dev": []} for d in code_distances}

    for p in p_list:
        print(f"\n--- p = {p:.4e} ---")
        for d in code_distances:
            pL, pL_dev = simulate_func(p, d, runtime_budget, p_graph=p)
            results[d]["p"].append(p)
            results[d]["pL"].append(pL)
            results[d]["pL_dev"].append(pL_dev)

    return results


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

def plot_comparison(results_soft, results_no_erasure,
                    code_distances, exposure, Pm, classes,
                    th_soft=None, th_no=None,
                    save_path="confusion_amb_threshold.pdf"):
    """Two-way comparison plot: soft erasure vs no-erasure."""
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = {3: 'C0', 5: 'C1', 7: 'C2', 9: 'C3', 11: 'C4', 13: 'C5', 15: 'C6'}

    for d in code_distances:
        clr = colors.get(d, 'gray')

        # No erasure (open circles, dashed)
        if d in results_no_erasure and len(results_no_erasure[d]["p"]) > 0:
            p_arr = np.array(results_no_erasure[d]["p"])
            pL_arr = np.array(results_no_erasure[d]["pL"])
            valid = pL_arr > 0
            ax.plot(p_arr[valid], pL_arr[valid], 'o--',
                    color=clr, markerfacecolor='white', markeredgecolor=clr,
                    markersize=6, linewidth=1.5)

        # Soft erasure (filled circles, solid)
        if d in results_soft and len(results_soft[d]["p"]) > 0:
            p_arr = np.array(results_soft[d]["p"])
            pL_arr = np.array(results_soft[d]["pL"])
            valid = pL_arr > 0
            ax.plot(p_arr[valid], pL_arr[valid], 'o-',
                    color=clr, markersize=6, linewidth=1.5,
                    label=f'd = {d}')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical error probability, $p$', fontsize=14)
    ax.set_ylabel('Logical error rate, $p_L$', fontsize=14)
    ax.set_xlim(1e-4, 2e-1)
    ax.set_ylim(1e-6, 1)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)

    # Threshold vertical lines
    y_pos = 1e-5
    if th_no is not None:
        ax.axvline(x=th_no, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        ax.text(th_no * 1.15, y_pos,
                f'$p_{{th}}={th_no*100:.2f}\\%$\n(no erasure)',
                fontsize=8, color='gray')
    if th_soft is not None:
        ax.axvline(x=th_soft, color='blue', linestyle='-', alpha=0.7, linewidth=2)
        ax.text(th_soft * 0.35, y_pos,
                f'$p_{{th}}={th_soft*100:.2f}\\%$\n(soft erasure)',
                fontsize=8, color='blue')

    # Line style legend
    ax.text(0.02, 0.18,
            'no erasure   (dashed, open)\n'
            'soft erasure  (solid, filled)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')

    sum_Rm = sum(c['Rm'] for c in classes)
    sum_Rc = sum(c['Rc'] for c in classes)
    max_wt = max(c['weight'] for c in classes) if classes else 1.0
    weight_str = ", ".join(f"{c['name']}={c['weight']/max_wt:.3f}" for c in classes)
    title = (f"Exposure={exposure}ms: Pm={Pm:.4f}, "
             f"Rm_total={sum_Rm:.4f}, Rc_total={sum_Rc:.4f}\n"
             f"(All amb zone → erasure, norm weights: {weight_str})")
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure: {save_path}")
    plt.show()


# ============== Main ==============

if __name__ == "__main__":
    # Default path for ambiguous zone CSVs
    default_amb_dir = os.path.join(
        os.path.expanduser("~"),
        "Documents/research/denoise/Noise2NoiseFlow/noise2noiseflow/uncertainty_weighted_outputs"
    )

    parser = argparse.ArgumentParser(
        description='Ambiguous Zone Erasure Threshold (soft weights)')
    parser.add_argument('--csv', required=True,
                        help='Confusion matrix CSV file (with Den/Amb rows)')
    parser.add_argument('--amb-dir', default=default_amb_dir,
                        help='Directory containing ambiguous_zone_data_Xms.csv files')
    parser.add_argument('--exposure', type=int, default=None,
                        help='Exposure time to use')
    parser.add_argument('--mode', choices=['params', 'quick', 'full', 'plot', 'lambda'],
                        default='params',
                        help='Mode: params (show only), quick, full, plot, lambda')
    parser.add_argument('--list-exposures', action='store_true',
                        help='List available exposures and exit')
    parser.add_argument('--max-half-weight', type=int, default=1,
                        help='UF decoder max_half_weight (default: 1)')
    parser.add_argument('--data-dir', default=None,
                        help='Directory for saving/loading result files')
    parser.add_argument('--output', default=None,
                        help='Output plot file path')
    args = parser.parse_args()

    # Parse confusion matrix CSV
    confusion_data = parse_confusion_csv(args.csv)

    if args.list_exposures:
        print(f"Available exposures: {sorted(confusion_data.keys())}")
        for exp in sorted(confusion_data.keys()):
            methods = sorted(confusion_data[exp].keys())
            amb_csv = os.path.join(args.amb_dir, f"ambiguous_zone_data_{exp}ms.csv")
            has_amb = os.path.exists(amb_csv)
            print(f"  exposure={exp}: methods={methods}, amb CSV={'OK' if has_amb else 'MISSING'}")
        sys.exit(0)

    if args.exposure is None:
        print("Available exposures:", sorted(confusion_data.keys()))
        print("Use --exposure <value> to select one, or --list-exposures for details")
        sys.exit(1)

    exp = args.exposure
    if exp not in confusion_data or 'Den' not in confusion_data[exp]:
        print(f"Error: Exposure {exp} not found or missing Den row")
        sys.exit(1)

    # Parse per-class ambiguous zone CSV
    amb_csv = os.path.join(args.amb_dir, f"ambiguous_zone_data_{exp}ms.csv")
    if not os.path.exists(amb_csv):
        print(f"Error: Amb zone CSV not found: {amb_csv}")
        sys.exit(1)

    amb_counts = parse_amb_zone_csv(amb_csv)
    if not amb_counts:
        print(f"Error: No weight class data in {amb_csv}")
        sys.exit(1)

    # Calculate parameters
    Pm, classes = calculate_amb_params(confusion_data, amb_counts, exp)
    print_params_summary(Pm, classes, exp, confusion_data)

    sum_Rm = sum(c['Rm'] for c in classes)
    sum_Rc = sum(c['Rc'] for c in classes)

    if args.mode == 'params':
        # Print the JSON config for reference
        soft_config = {
            "measurement_error_rate_total": Pm,
            "erasure_classes": [
                {"Rm": round(c['Rm'], 8), "Rc": round(c['Rc'], 8)}
                for c in classes
            ],
        }
        print(f"\n  Soft (multi-class) config:")
        print(f"  {json.dumps(soft_config, indent=4)}")
        sys.exit(0)

    # Setup directories
    data_dir = args.data_dir or f"results_amb_exp{exp}"
    os.makedirs(data_dir, exist_ok=True)
    output = args.output or os.path.join(data_dir, f"amb_threshold_exp{exp}.pdf")

    compile_code_if_necessary()

    mhw = args.max_half_weight

    if args.mode in ('quick', 'full'):
        if args.mode == 'quick':
            code_distances = [5, 7, 9, 11]
            runtime_budget = (300, 45)
            p_sweep = np.logspace(-4, -1, 12)
        else:  # full
            code_distances = [5, 7, 9, 11, 13]
            runtime_budget = (1000, 180)
            p_sweep = np.logspace(-4, -1, 20)

        p_list = p_sweep.tolist()

        # ========== 1/2: Soft weighted erasure ==========
        print(f"\n{'='*60}")
        print(f" [1/2] Soft weighted erasure (max_half_weight={mhw})")
        max_wt = max(c['weight'] for c in classes) if classes else 1.0
        weights_str = ", ".join(f"{c['name']}={c['weight']/max_wt:.3f}" for c in classes)
        print(f"       Norm weights: {weights_str}")
        print(f"{'='*60}")
        sim_soft = create_simulate_func_soft(Pm, classes, max_half_weight=mhw)
        results_soft = run_p_sweep(sim_soft, code_distances, p_list, runtime_budget)
        save_results(results_soft,
                     {"exposure": exp, "Pm": Pm,
                      "classes": [{"name": c["name"], "Rm": c["Rm"], "Rc": c["Rc"], "weight": c["weight"]}
                                  for c in classes],
                      "type": "soft", "max_half_weight": mhw},
                     os.path.join(data_dir, "results_soft.json"))

        # ========== 2/2: No erasure ==========
        print(f"\n{'='*60}")
        print(f" [2/2] No erasure (Pm={Pm:.6f}, pure measurement error)")
        print(f"{'='*60}")
        sim_no = create_simulate_func_no_erasure(Pm, max_half_weight=mhw)
        results_no = run_p_sweep(sim_no, code_distances, p_list, runtime_budget)
        save_results(results_no,
                     {"exposure": exp, "Pm": Pm,
                      "type": "no_erasure", "max_half_weight": mhw},
                     os.path.join(data_dir, "results_no_erasure.json"))

        # ========== Λ-factor ==========
        lambda_soft = compute_lambda_factor(results_soft, code_distances)
        lambda_no = compute_lambda_factor(results_no, code_distances)
        print_lambda_summary(lambda_soft, label="Soft weighted erasure")
        print_lambda_summary(lambda_no, label="No erasure (baseline)")

        # ========== Plot ==========
        plot_comparison(results_soft, results_no,
                        code_distances, exp, Pm, classes, save_path=output)
        lambda_path = output.replace('.pdf', '_lambda.pdf')
        lambda_datasets = [
            ('soft erasure', 'C0', 'o', '-', lambda_soft),
            ('no erasure',   'gray', 's', '--', lambda_no),
        ]
        plot_lambda_comparison(
            lambda_datasets, code_distances,
            title=f'Λ-factor: Exposure={exp}ms, Pm={Pm:.4f}\n(Λ>1 = error suppression working)',
            save_path=lambda_path)

    elif args.mode == 'lambda':
        # Load pre-computed results and compute Λ
        results_soft, _ = load_results(os.path.join(data_dir, "results_soft.json"))
        results_no, _ = load_results(os.path.join(data_dir, "results_no_erasure.json"))
        code_distances = sorted(results_soft.keys())

        lambda_soft = compute_lambda_factor(results_soft, code_distances)
        lambda_no = compute_lambda_factor(results_no, code_distances)

        print_lambda_summary(lambda_soft, label="Soft weighted erasure")
        print_lambda_summary(lambda_no, label="No erasure (baseline)")

        lambda_path = output.replace('.pdf', '_lambda.pdf')
        lambda_datasets = [
            ('soft erasure', 'C0', 'o', '-', lambda_soft),
            ('no erasure',   'gray', 's', '--', lambda_no),
        ]
        plot_lambda_comparison(
            lambda_datasets, code_distances,
            title=f'Λ-factor: Exposure={exp}ms, Pm={Pm:.4f}\n(Λ>1 = error suppression working)',
            save_path=lambda_path)

    elif args.mode == 'plot':
        # Load pre-computed results
        results_soft, _ = load_results(os.path.join(data_dir, "results_soft.json"))
        results_no, _ = load_results(os.path.join(data_dir, "results_no_erasure.json"))
        code_distances = sorted(results_soft.keys())

        # Estimate thresholds
        print("\n>>> Soft erasure: threshold estimate...")
        th_soft, _ = estimate_threshold_from_data(results_soft, code_distances, verbose=True)
        print("\n>>> No erasure: threshold estimate...")
        th_no, _ = estimate_threshold_from_data(results_no, code_distances, verbose=True)

        # Λ-factor
        lambda_soft = compute_lambda_factor(results_soft, code_distances)
        lambda_no = compute_lambda_factor(results_no, code_distances)
        print_lambda_summary(lambda_soft, label="Soft weighted erasure")
        print_lambda_summary(lambda_no, label="No erasure (baseline)")

        plot_comparison(results_soft, results_no,
                        code_distances, exp, Pm, classes,
                        th_soft=th_soft, th_no=th_no,
                        save_path=output)
        lambda_path = output.replace('.pdf', '_lambda.pdf')
        lambda_datasets = [
            ('soft erasure', 'C0', 'o', '-', lambda_soft),
            ('no erasure',   'gray', 's', '--', lambda_no),
        ]
        plot_lambda_comparison(
            lambda_datasets, code_distances,
            title=f'Λ-factor: Exposure={exp}ms, Pm={Pm:.4f}\n(Λ>1 = error suppression working)',
            save_path=lambda_path)
