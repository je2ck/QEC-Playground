"""
Threshold vs Measurement Error Probability

x축: measurement error probability Pm (0 ~ 0.03)
y축: threshold physical error rate p_th

각 Pm 값에 대해 ThresholdAnalyzer (rough → precise curve fitting)로
threshold를 추정한 뒤, 이를 그래프로 나타냅니다.

Re=0 (Rm=0, Rc=0): measurement erasure 없음

사용법:
    python threshold_vs_measurement_error.py --mode quick
    python threshold_vs_measurement_error.py --mode full
    python threshold_vs_measurement_error.py --mode plot
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# QEC Playground 경로 설정
import subprocess
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

from utils import ProgressTracker, resolve_parallel_workers
from concurrent.futures import ProcessPoolExecutor, as_completed


# ============== 시뮬레이션 함수 ==============

def create_simulate_func(Pm, Rm=0.0, Rc=0.0):
    """
    Measurement erasure 모델에 대한 simulate_func 생성 (ThresholdAnalyzer 호환)

    Args:
        Pm: total measurement error probability
        Rm: erasure rate on measurement error (0~1), default 0
        Rc: erasure rate on correct measurement (0~1), default 0
    """
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

        print(f"  [Pm={Pm:.4f}] d={d:2d}, p={p:.4e}: pL={pL:.4e} ± {pL_dev:.2e}")
        return (pL, pL_dev)

    return simulate_func


# ============== ThresholdAnalyzer 래퍼 ==============

def find_threshold_for_Pm(Pm, Rm, Rc, code_distances, rough_code_distances,
                          rough_runtime_budget, runtime_budget,
                          verbose=True):
    """
    하나의 Pm 값에 대해 ThresholdAnalyzer로 threshold 추정
    (rough estimate → precise estimate)

    Returns:
        threshold: 추정된 threshold (없으면 None)
        threshold_err: threshold 오차 (없으면 None)
    """
    print(f"\n{'='*60}")
    print(f"  Finding threshold for Pm = {Pm:.4f}")
    print(f"{'='*60}")

    simulate_func = create_simulate_func(Pm, Rm, Rc)

    analyzer = ThresholdAnalyzer(
        code_distances=code_distances,
        simulate_func=simulate_func,
        default_rough_runtime_budget=rough_runtime_budget,
        default_runtime_budget=runtime_budget,
    )

    analyzer.rough_code_distances = rough_code_distances
    analyzer.rough_runtime_budgets = [rough_runtime_budget] * len(rough_code_distances)
    analyzer.verbose = verbose

    # threshold가 ~0.5~1% 정도이므로 시작점을 낮게
    analyzer.rough_init_search_start_p = 0.05

    try:
        # rough → precise
        rough_popt, rough_perr = analyzer.rough_estimate()
        rough_th = rough_popt[3]
        print(f"    [Rough] Pm={Pm:.4f} → p_th ≈ {rough_th:.6f} ({rough_th*100:.3f}%)")

        popt, perr = analyzer.precise_estimate(rough_popt)
        threshold = popt[3]
        threshold_err = perr[3]

        # retry if error is too large
        if threshold_err / threshold > 0.01:
            print(f"    [Retry] Error too large ({threshold_err/threshold:.1%}), retrying...")
            popt, perr = analyzer.precise_estimate(popt)
            threshold = popt[3]
            threshold_err = perr[3]

        print(f"    ★ Pm={Pm:.4f} → p_th = {threshold:.6f} ({threshold*100:.3f}%) ± {threshold_err:.6f}")
        return threshold, threshold_err

    except Exception as e:
        print(f"    ✗ Pm={Pm:.4f} → threshold estimation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ============== 저장/로드 ==============

def save_results(data, filename):
    """결과를 JSON으로 저장"""
    serializable = {
        "Pm_list": data["Pm_list"],
        "thresholds": data["thresholds"],
        "threshold_errs": data["threshold_errs"],
        "params": data["params"],
    }
    with open(filename, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved results to: {filename}")


def load_results(filename):
    """JSON에서 결과 로드"""
    with open(filename, 'r') as f:
        return json.load(f)


# ============== 플롯 ==============

def plot_threshold_vs_Pm(Pm_list, thresholds, threshold_errs=None,
                         save_path="threshold_vs_Pm.pdf",
                         title="Threshold vs Measurement Error Probability (Re=0)"):
    """
    x축: Pm (measurement error probability)
    y축: p_th (threshold physical error rate, %)
    """
    valid = [(Pm, th, err) for Pm, th, err in
             zip(Pm_list, thresholds,
                 threshold_errs if threshold_errs else [None]*len(Pm_list))
             if th is not None]

    if not valid:
        print("No valid threshold data to plot!")
        return

    Pm_valid = [v[0] for v in valid]
    th_valid = [v[1] for v in valid]

    fig, ax = plt.subplots(figsize=(8, 6))

    th_valid_pct = [t * 100 for t in th_valid]

    ax.plot(Pm_valid, th_valid_pct, 'o-', color='C0', markersize=7,
            linewidth=2)

    ax.set_xlabel('Measurement error probability, $P_m$', fontsize=14)
    ax.set_ylabel('Threshold physical error rate, $p_{th}$ (%)', fontsize=14)
    ax.set_title(title, fontsize=13)

    ax.set_xlim(-0.001, max(Pm_valid) * 1.05 + 0.001)
    y_min = min(th_valid_pct) * 0.9
    y_max = max(th_valid_pct) * 1.1
    ax.set_ylim(y_min, y_max)

    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)

    for Pm, th_pct in zip(Pm_valid, th_valid_pct):
        ax.annotate(f'{th_pct:.2f}%',
                    xy=(Pm, th_pct), xytext=(5, 10),
                    textcoords='offset points', fontsize=8,
                    color='C0', ha='left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {save_path}")
    plt.show()


# ============== 메인 ==============

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Threshold vs Measurement Error Probability (Re=0)')
    parser.add_argument('--mode', choices=['quick', 'full', 'plot'],
                        default='quick', help='Simulation mode')
    parser.add_argument('--output', default='threshold_vs_Pm.pdf',
                        help='Output figure path')
    parser.add_argument('--data-dir', default='results_threshold_vs_Pm',
                        help='Directory for data files')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers for Pm values (0 = all cores, 1 = sequential)')
    args = parser.parse_args()
    args.parallel = resolve_parallel_workers(args.parallel)

    os.makedirs(args.data_dir, exist_ok=True)
    if not os.path.isabs(args.output):
        args.output = os.path.join(args.data_dir, args.output)
    compile_code_if_necessary()

    # Re=0: Rm=0, Rc=0 (no erasure conversion)
    Rm = 0.0
    Rc = 0.0

    if args.mode == 'quick':
        print("=== Quick mode: Threshold vs Pm (Re=0) ===")
        print(f"    Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ----- 설정 -----
        Pm_list = np.linspace(0, 0.03, 13).tolist()
        rough_code_distances = [5, 7]
        code_distances = [5, 7, 9, 11]
        rough_runtime_budget = (1000, 60)    # rough: 빠르게
        runtime_budget = (3000, 120)          # precise: 좀 더 통계

        print(f"    Pm values: {[f'{Pm:.4f}' for Pm in Pm_list]}")
        print(f"    Rough code distances: {rough_code_distances}")
        print(f"    Precise code distances: {code_distances}")

        thresholds = [None] * len(Pm_list)
        threshold_errs = [None] * len(Pm_list)

        if args.parallel > 1:
            print(f"    Parallel workers: {args.parallel}")
            with ProcessPoolExecutor(max_workers=args.parallel) as executor:
                futures = {}
                for idx, Pm in enumerate(Pm_list):
                    fut = executor.submit(
                        find_threshold_for_Pm,
                        Pm, Rm, Rc, code_distances, rough_code_distances,
                        rough_runtime_budget, runtime_budget,
                    )
                    futures[fut] = idx
                for fut in as_completed(futures):
                    idx = futures[fut]
                    th, th_err = fut.result()
                    thresholds[idx] = th
                    threshold_errs[idx] = th_err
                    print(f"    [{idx+1}/{len(Pm_list)}] Pm={Pm_list[idx]:.4f} done")
        else:
            pm_tracker = ProgressTracker(len(Pm_list), "Pm values", print_every=1)
            for idx, Pm in enumerate(Pm_list):
                pm_tracker.begin_task()
                th, th_err = find_threshold_for_Pm(
                    Pm, Rm, Rc, code_distances, rough_code_distances,
                    rough_runtime_budget, runtime_budget,
                )
                thresholds[idx] = th
                threshold_errs[idx] = th_err
                pm_tracker.end_task()
            pm_tracker.summary()

        all_data = {
            "Pm_list": [float(x) for x in Pm_list],
            "thresholds": [float(t) if t is not None else None for t in thresholds],
            "threshold_errs": [float(e) if e is not None else None for e in threshold_errs],
            "params": {"Rm": Rm, "Rc": Rc,
                       "rough_code_distances": rough_code_distances,
                       "code_distances": code_distances,
                       "mode": "quick"},
        }
        save_results(all_data, os.path.join(args.data_dir, "results_quick.json"))

        # 결과 요약
        print(f"\n{'='*60}")
        print(f"  Summary: Threshold vs Pm (Re=0)")
        print(f"{'='*60}")
        print(f"  {'Pm':>8s}  {'p_th':>12s}  {'p_th (%)':>10s}  {'± err':>10s}")
        print(f"  {'-'*44}")
        for Pm, th, err in zip(Pm_list, thresholds, threshold_errs):
            if th is not None:
                err_str = f"{err*100:.4f}%" if err else "N/A"
                print(f"  {Pm:>8.4f}  {th:>12.6f}  {th*100:>9.3f}%  {err_str:>10s}")
            else:
                print(f"  {Pm:>8.4f}  {'N/A':>12s}  {'N/A':>10s}  {'N/A':>10s}")

        # 그래프
        plot_threshold_vs_Pm(Pm_list, thresholds, threshold_errs,
                             save_path=args.output)

        print(f"\n    End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    elif args.mode == 'full':
        print("=== Full mode: Threshold vs Pm (Re=0) ===")
        print(f"    Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        Pm_list = np.linspace(0, 0.03, 13).tolist()
        rough_code_distances = [5, 7]
        code_distances = [7, 9, 11, 13]
        rough_runtime_budget = (3000, 120)
        runtime_budget = (18000, 3600)

        print(f"    Pm values: {[f'{Pm:.4f}' for Pm in Pm_list]}")
        print(f"    Rough code distances: {rough_code_distances}")
        print(f"    Precise code distances: {code_distances}")

        thresholds = [None] * len(Pm_list)
        threshold_errs = [None] * len(Pm_list)

        if args.parallel > 1:
            print(f"    Parallel workers: {args.parallel}")
            with ProcessPoolExecutor(max_workers=args.parallel) as executor:
                futures = {}
                for idx, Pm in enumerate(Pm_list):
                    fut = executor.submit(
                        find_threshold_for_Pm,
                        Pm, Rm, Rc, code_distances, rough_code_distances,
                        rough_runtime_budget, runtime_budget,
                    )
                    futures[fut] = idx
                for fut in as_completed(futures):
                    idx = futures[fut]
                    th, th_err = fut.result()
                    thresholds[idx] = th
                    threshold_errs[idx] = th_err
                    print(f"    [{idx+1}/{len(Pm_list)}] Pm={Pm_list[idx]:.4f} done")
        else:
            pm_tracker = ProgressTracker(len(Pm_list), "Pm values", print_every=1)
            for idx, Pm in enumerate(Pm_list):
                pm_tracker.begin_task()
                th, th_err = find_threshold_for_Pm(
                    Pm, Rm, Rc, code_distances, rough_code_distances,
                    rough_runtime_budget, runtime_budget,
                )
                thresholds[idx] = th
                threshold_errs[idx] = th_err
                pm_tracker.end_task()
            pm_tracker.summary()

        all_data = {
            "Pm_list": Pm_list,
            "thresholds": [float(t) if t is not None else None for t in thresholds],
            "threshold_errs": [float(e) if e is not None else None for e in threshold_errs],
            "params": {"Rm": Rm, "Rc": Rc,
                       "rough_code_distances": rough_code_distances,
                       "code_distances": code_distances,
                       "mode": "full"},
        }
        save_results(all_data, os.path.join(args.data_dir, "results_full.json"))

        print(f"\n{'='*60}")
        print(f"  Summary: Threshold vs Pm (Re=0)")
        print(f"{'='*60}")
        print(f"  {'Pm':>8s}  {'p_th':>12s}  {'p_th (%)':>10s}  {'± err':>10s}")
        print(f"  {'-'*44}")
        for Pm, th, err in zip(Pm_list, thresholds, threshold_errs):
            if th is not None:
                err_str = f"{err*100:.4f}%" if err else "N/A"
                print(f"  {Pm:>8.4f}  {th:>12.6f}  {th*100:>9.3f}%  {err_str:>10s}")
            else:
                print(f"  {Pm:>8.4f}  {'N/A':>12s}  {'N/A':>10s}  {'N/A':>10s}")

        plot_threshold_vs_Pm(Pm_list, thresholds, threshold_errs,
                             save_path=args.output)

        print(f"\n    End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    elif args.mode == 'plot':
        print("=== Plot mode: loading saved results ===")

        for fname in ["results_full.json", "results_quick.json"]:
            fpath = os.path.join(args.data_dir, fname)
            if os.path.exists(fpath):
                print(f"Loading: {fpath}")
                all_data = load_results(fpath)
                break
        else:
            print(f"No saved results found in {args.data_dir}/")
            sys.exit(1)

        Pm_list = all_data["Pm_list"]
        thresholds = all_data["thresholds"]
        threshold_errs = all_data["threshold_errs"]

        print(f"\n{'='*60}")
        print(f"  Summary: Threshold vs Pm (Re=0)")
        print(f"{'='*60}")
        print(f"  {'Pm':>8s}  {'p_th':>12s}  {'p_th (%)':>10s}")
        print(f"  {'-'*34}")
        for Pm, th in zip(Pm_list, thresholds):
            if th is not None:
                print(f"  {Pm:>8.4f}  {th:>12.6f}  {th*100:>9.3f}%")
            else:
                print(f"  {Pm:>8.4f}  {'N/A':>12s}  {'N/A':>10s}")

        plot_threshold_vs_Pm(Pm_list, thresholds, threshold_errs,
                             save_path=args.output)
