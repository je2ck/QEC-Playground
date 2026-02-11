"""
Confusion Matrix → Measurement Erasure Threshold Analysis

실험에서 얻은 confusion matrix CSV 파일로부터 Pm, Rm, Rc를 자동 계산하고,
해당 파라미터로 measurement erasure threshold 시뮬레이션을 수행합니다.

Confusion matrix 해석:
  - Den (Denoised) 행: measurement 정확도
    - TP: syndrome=1이고 측정도 1 (정확)
    - TN: syndrome=0이고 측정도 0 (정확)
    - FP: syndrome=0인데 측정이 1 (measurement error)
    - FN: syndrome=1인데 측정이 0 (measurement error)
    → Pm = (FP + FN) / Total

  - E (Erasure) 행: erasure로 분류된 측정의 confusion
    - E의 FP + FN = erasure 중 measurement error가 있었던 수
    - E의 TP + TN = erasure 중 measurement error가 없었던 수
    → Rm = (E_FP + E_FN) / (Den_FP + Den_FN)   = P(erasure | error)
    → Rc = (E_TP + E_TN) / (Den_TP + Den_TN)   = P(erasure | no error)

사용법:
    python confusion_threshold.py --csv confusion_amb.csv --exposure 8 --mode quick
    python confusion_threshold.py --csv confusion_amb.csv --exposure 8 --mode full
    python confusion_threshold.py --csv confusion_amb.csv --exposure 8 --mode plot
    python confusion_threshold.py --csv confusion_amb.csv --list-exposures
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import subprocess

# QEC Playground 경로 설정
qec_playground_root_dir = subprocess.run(
    "git rev-parse --show-toplevel",
    cwd=os.path.dirname(os.path.abspath(__file__)),
    shell=True, check=True, capture_output=True
).stdout.decode(sys.stdout.encoding).strip(" \r\n")
sys.path.insert(0, os.path.join(qec_playground_root_dir, "benchmark", "threshold_analyzer"))

from threshold_analyzer import (
    ThresholdAnalyzer,
    qecp_benchmark_simulate_func_command_vec,
    run_qecp_command_get_stdout,
    compile_code_if_necessary,
)
from utils import find_crossing_point, estimate_threshold_from_data, merge_results, ProgressTracker, run_parallel_simulations, scaled_runtime_budget, resolve_parallel_workers


# ============== CSV 파싱 ==============

def parse_confusion_csv(csv_path):
    """
    Confusion matrix CSV 파일을 파싱합니다.
    
    Returns:
        {exposure: {method: {"TP": int, "TN": int, "FP": int, "FN": int, ...}}}
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


def calculate_params(confusion_data, exposure):
    """
    특정 exposure의 Den과 E 행에서 Pm, Rm, Rc를 계산합니다.
    
    Args:
        confusion_data: parse_confusion_csv의 결과
        exposure: 선택할 exposure 값
    
    Returns:
        (Pm, Rm, Rc) 튜플
    """
    if exposure not in confusion_data:
        raise ValueError(f"Exposure {exposure} not found. Available: {sorted(confusion_data.keys())}")
    
    exp_data = confusion_data[exposure]
    
    if 'Den' not in exp_data:
        raise ValueError(f"'Den' method not found for exposure {exposure}")
    if 'E' not in exp_data:
        raise ValueError(f"'E' method not found for exposure {exposure}")
    
    den = exp_data['Den']
    e = exp_data['E']
    
    # Den에서 Pm 계산
    den_total = den['TP'] + den['TN'] + den['FP'] + den['FN']
    den_errors = den['FP'] + den['FN']       # measurement errors
    den_correct = den['TP'] + den['TN']       # correct measurements
    Pm = den_errors / den_total
    
    # E에서 Rm, Rc 계산
    e_errors = e['FP'] + e['FN']      # erasure 중 measurement error
    e_correct = e['TP'] + e['TN']     # erasure 중 correct measurement
    
    Rm = e_errors / den_errors if den_errors > 0 else 0.0       # P(erasure | error)
    Rc = e_correct / den_correct if den_correct > 0 else 0.0    # P(erasure | no error)
    
    return Pm, Rm, Rc


def print_params_summary(confusion_data, exposure):
    """파라미터 계산 결과를 출력합니다."""
    Pm, Rm, Rc = calculate_params(confusion_data, exposure)
    
    den = confusion_data[exposure]['Den']
    e = confusion_data[exposure]['E']
    
    den_total = den['TP'] + den['TN'] + den['FP'] + den['FN']
    den_errors = den['FP'] + den['FN']
    den_correct = den['TP'] + den['TN']
    e_total = e['TP'] + e['TN'] + e['FP'] + e['FN']
    e_errors = e['FP'] + e['FN']
    e_correct = e['TP'] + e['TN']
    
    print(f"\n{'='*60}")
    print(f" Exposure = {exposure}")
    print(f"{'='*60}")
    print(f" Den: TP={den['TP']}, TN={den['TN']}, FP={den['FP']}, FN={den['FN']}  (total={den_total})")
    print(f"   E: TP={e['TP']}, TN={e['TN']}, FP={e['FP']}, FN={e['FN']}  (total={e_total})")
    print(f"")
    print(f" Pm  = {den_errors}/{den_total} = {Pm:.6f}  (measurement error rate)")
    print(f" Rm  = {e_errors}/{den_errors} = {Rm:.6f}  (P(erasure | error))")
    print(f" Rc  = {e_correct}/{den_correct} = {Rc:.6f}  (P(erasure | no error))")
    print(f"")
    
    # Noise model parameters
    meas_err_rate = Pm * (1 - Rm)
    meas_err_with_erasure = Pm * Rm
    meas_erasure_no_error = (1 - Pm) * Rc
    total_erasure = meas_err_with_erasure + meas_erasure_no_error
    
    print(f" Derived noise model parameters:")
    print(f"   measurement_error_rate            = Pm*(1-Rm) = {meas_err_rate:.6f}  (hidden Pauli)")
    print(f"   measurement_error_rate_with_erasure = Pm*Rm   = {meas_err_with_erasure:.6f}  (erasure + error)")
    print(f"   measurement_erasure_rate_no_error   = (1-Pm)*Rc = {meas_erasure_no_error:.6f}  (erasure only)")
    print(f"   total erasure rate                            = {total_erasure:.6f}")
    if total_erasure > 0:
        p_err_given_erasure = meas_err_with_erasure / total_erasure
        print(f"   P(error | erasure)                            = {p_err_given_erasure:.6f}")
    print(f"{'='*60}")
    
    return Pm, Rm, Rc


# ============== 시뮬레이션 함수 ==============

def create_simulate_func(Pm, Rm, Rc=0.0):
    """Measurement erasure 모델에 대한 simulate_func 생성"""
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
            p_graph=p_graph
        )
        
        stdout, returncode = run_qecp_command_get_stdout(command)
        if returncode != 0:
            print(f"[ERROR] simulation failed for p={p}, d={d}")
            return (0.5, 1.0)
        
        full_result = stdout.strip(" \r\n").split("\n")[-1]
        lst = full_result.split(" ")
        pL = float(lst[5])
        pL_dev = float(lst[7])
        
        print(f"  [exp] d={d:2d}, p={p:.4e}: pL={pL:.4e} ± {pL_dev:.2e}")
        return (pL, pL_dev)
    
    return simulate_func


def create_simulate_func_no_erasure(Pm):
    """Erasure 없이 순수 measurement error만 사용하는 simulate_func (비교용)"""
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
            p_graph=p_graph
        )
        
        stdout, returncode = run_qecp_command_get_stdout(command)
        if returncode != 0:
            print(f"[ERROR] simulation failed for p={p}, d={d}")
            return (0.5, 1.0)
        
        full_result = stdout.strip(" \r\n").split("\n")[-1]
        lst = full_result.split(" ")
        pL = float(lst[5])
        pL_dev = float(lst[7])
        
        print(f"  [no-erasure] d={d:2d}, p={p:.4e}: pL={pL:.4e} ± {pL_dev:.2e}")
        return (pL, pL_dev)
    
    return simulate_func


def run_p_sweep(simulate_func, code_distances, p_list, runtime_budget, n_workers=1):
    """고정된 p 값들에 대해 시뮬레이션 수행"""
    if n_workers > 1:
        return run_parallel_simulations(simulate_func, code_distances, p_list, runtime_budget, n_workers)

    results = {d: {"p": [], "pL": [], "pL_dev": []} for d in code_distances}

    total_sims = len(p_list) * len(code_distances)
    tracker = ProgressTracker(total_sims, "simulations", print_every=len(code_distances))

    d_base = min(code_distances)
    for p in p_list:
        print(f"\n--- p = {p:.4e} ---")
        for d in code_distances:
            tracker.begin_task()
            pL, pL_dev = simulate_func(p, d, scaled_runtime_budget(runtime_budget, d, d_base), p_graph=p)
            results[d]["p"].append(p)
            results[d]["pL"].append(pL)
            results[d]["pL_dev"].append(pL_dev)
            tracker.end_task()

    tracker.summary()
    return results


# ============== 플롯 함수 ==============

def plot_comparison(results_with_erasure, results_no_erasure, code_distances,
                    exposure, Pm, Rm, Rc,
                    threshold_with=None, threshold_without=None,
                    save_path="confusion_threshold.pdf"):
    """Erasure 있을 때 / 없을 때 비교 그래프"""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    colors = {3: 'C0', 5: 'C1', 7: 'C2', 9: 'C3', 11: 'C4', 13: 'C5', 15: 'C6'}
    
    for d in code_distances:
        color = colors.get(d, 'gray')
        
        # No erasure (open circles, dashed)
        if d in results_no_erasure and len(results_no_erasure[d]["p"]) > 0:
            p_arr = np.array(results_no_erasure[d]["p"])
            pL_arr = np.array(results_no_erasure[d]["pL"])
            valid = pL_arr > 0
            ax.plot(p_arr[valid], pL_arr[valid], 'o--',
                    color=color, markerfacecolor='white', markeredgecolor=color,
                    markersize=6, linewidth=1.5)
        
        # With erasure (filled circles, solid)
        if d in results_with_erasure and len(results_with_erasure[d]["p"]) > 0:
            p_arr = np.array(results_with_erasure[d]["p"])
            pL_arr = np.array(results_with_erasure[d]["pL"])
            valid = pL_arr > 0
            ax.plot(p_arr[valid], pL_arr[valid], 'o-',
                    color=color, markersize=6, linewidth=1.5,
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
    
    if threshold_without is not None:
        ax.axvline(x=threshold_without, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        ax.text(threshold_without*1.1, 1e-5,
                f'$p_{{th}}={threshold_without*100:.2f}\\%$\n(no erasure)', fontsize=9, color='gray')
    if threshold_with is not None:
        ax.axvline(x=threshold_with, color='blue', linestyle=':', alpha=0.7, linewidth=2)
        ax.text(threshold_with*0.5, 1e-5,
                f'$p_{{th}}={threshold_with*100:.2f}\\%$\n(with erasure)', fontsize=9, color='blue')
    
    ax.text(0.02, 0.18, 'no erasure\n(dashed, open)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.02, 0.08, 'with erasure\n(solid, filled)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    
    title = (f"Exposure={exposure}: Pm={Pm:.4f}, Rm={Rm:.4f}, Rc={Rc:.6f}")
    ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {save_path}")
    plt.show()


def save_results(results, params, filename):
    """결과를 JSON으로 저장"""
    data = {"params": params, "results": {}}
    for d, vals in results.items():
        data["results"][str(d)] = {
            "p": [float(x) for x in vals["p"]],
            "pL": [float(x) for x in vals["pL"]],
            "pL_dev": [float(x) for x in vals["pL_dev"]]
        }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to: {filename}")


def load_results(filename):
    """JSON에서 결과 로드"""
    with open(filename, 'r') as f:
        data = json.load(f)
    params = data.get("params", {})
    results = {}
    for d_str, vals in data["results"].items():
        results[int(d_str)] = vals
    return results, params


# ============== 메인 ==============

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Confusion Matrix → Measurement Erasure Threshold')
    parser.add_argument('--csv', required=True, help='Confusion matrix CSV file path')
    parser.add_argument('--exposure', type=int, default=None, help='Exposure time to use')
    parser.add_argument('--mode', choices=['quick', 'full', 'plot', 'params'], default='params',
                        help='Mode: params (show only), quick, full, plot')
    parser.add_argument('--list-exposures', action='store_true', help='List available exposures')
    parser.add_argument('--data-dir', default=None, help='Directory for data files')
    parser.add_argument('--output', default=None, help='Output plot file path')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers (0 = all cores, 1 = sequential)')
    args = parser.parse_args()
    args.parallel = resolve_parallel_workers(args.parallel)
    
    # CSV 파싱
    confusion_data = parse_confusion_csv(args.csv)
    
    if args.list_exposures:
        print(f"Available exposures: {sorted(confusion_data.keys())}")
        for exp in sorted(confusion_data.keys()):
            methods = sorted(confusion_data[exp].keys())
            print(f"  exposure={exp}: methods={methods}")
        sys.exit(0)
    
    if args.exposure is None:
        print("Available exposures:", sorted(confusion_data.keys()))
        print("Use --exposure <value> to select one, or --list-exposures for details")
        sys.exit(1)
    
    # 파라미터 계산 및 출력
    Pm, Rm, Rc = print_params_summary(confusion_data, args.exposure)
    
    if args.mode == 'params':
        sys.exit(0)
    
    # 데이터 디렉토리 설정
    data_dir = args.data_dir or f"results_confusion_exp{args.exposure}"
    os.makedirs(data_dir, exist_ok=True)
    output = args.output or os.path.join(data_dir, f"confusion_threshold_exp{args.exposure}.pdf")
    
    compile_code_if_necessary()
    
    if args.mode in ('quick', 'full'):
        if args.mode == 'quick':
            code_distances = [5, 7, 9, 11]
            runtime_budget = (300, 45)
            p_sweep = np.logspace(-4, -1, 12)
        else:
            code_distances = [5, 7, 9, 11, 13]
            runtime_budget = (40000, 3600)
            p_sweep = np.logspace(-4, -1, 20)
        
        p_list = p_sweep.tolist()
        
        # ========== With erasure (from confusion matrix) ==========
        print(f"\n{'='*60}")
        print(f" With measurement erasure (Pm={Pm:.4f}, Rm={Rm:.4f}, Rc={Rc:.6f})")
        print(f"{'='*60}")
        sim_with = create_simulate_func(Pm, Rm, Rc)
        results_with = run_p_sweep(sim_with, code_distances, p_list, runtime_budget, n_workers=args.parallel)
        save_results(results_with,
                     {"exposure": args.exposure, "Pm": Pm, "Rm": Rm, "Rc": Rc, "type": "with_erasure"},
                     os.path.join(data_dir, "results_with_erasure.json"))
        
        # ========== Without erasure (pure measurement error, for comparison) ==========
        print(f"\n{'='*60}")
        print(f" Without erasure (Pm={Pm:.4f}, pure measurement error)")
        print(f"{'='*60}")
        sim_without = create_simulate_func_no_erasure(Pm)
        results_without = run_p_sweep(sim_without, code_distances, p_list, runtime_budget, n_workers=args.parallel)
        save_results(results_without,
                     {"exposure": args.exposure, "Pm": Pm, "type": "no_erasure"},
                     os.path.join(data_dir, "results_no_erasure.json"))
        
        # Plot
        plot_comparison(results_with, results_without, code_distances,
                        args.exposure, Pm, Rm, Rc, save_path=output)
    
    elif args.mode == 'plot':
        results_with, params_with = load_results(os.path.join(data_dir, "results_with_erasure.json"))
        results_without, _ = load_results(os.path.join(data_dir, "results_no_erasure.json"))
        code_distances = sorted(results_with.keys())
        
        print("\n>>> With erasure: threshold from crossing points...")
        th_with, _ = estimate_threshold_from_data(results_with, code_distances, verbose=True)
        print("\n>>> Without erasure: threshold from crossing points...")
        th_without, _ = estimate_threshold_from_data(results_without, code_distances, verbose=True)
        
        plot_comparison(results_with, results_without, code_distances,
                        args.exposure, Pm, Rm, Rc,
                        threshold_with=th_with, threshold_without=th_without,
                        save_path=output)
