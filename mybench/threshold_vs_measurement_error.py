"""
Threshold vs Measurement Error Probability

x축: measurement error probability Pm (0 ~ 0.03)
y축: threshold physical error rate p_th

각 Pm 값에 대해 여러 code distance에서 pL(p) 시뮬레이션을 수행하고,
crossing point로 threshold를 추정한 뒤, 이를 그래프로 나타냅니다.

Re=0 (Rm=0, Rc=0): measurement erasure 없음

사용법:
    python threshold_vs_measurement_error.py --mode quick
    python threshold_vs_measurement_error.py --mode full
    python threshold_vs_measurement_error.py --mode plot
"""

import os
import sys
import json
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
    qecp_benchmark_simulate_func_command_vec,
    run_qecp_command_get_stdout,
    compile_code_if_necessary,
)

from utils import find_crossing_point, estimate_threshold_from_data, ProgressTracker, run_parallel_simulations, scaled_runtime_budget, resolve_parallel_workers, run_p_sweep_with_checkpoint, clean_checkpoints


# ============== 시뮬레이션 함수 ==============

def create_simulate_func(Pm, Rm=0.0, Rc=0.0):
    """
    Measurement erasure 모델에 대한 simulate_func 생성
    
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


def run_p_sweep(Pm, Rm, Rc, code_distances, p_list, runtime_budget, n_workers=1, checkpoint_path=None):
    """고정된 Pm에서 p 값들에 대해 시뮬레이션 수행 (checkpoint 지원)"""
    print(f"\n>>> Running p-sweep for Pm={Pm}, Rm={Rm}, Rc={Rc}")

    simulate_func = create_simulate_func(Pm, Rm, Rc)

    return run_p_sweep_with_checkpoint(
        simulate_func, code_distances, p_list, runtime_budget,
        checkpoint_path=checkpoint_path, n_workers=n_workers)


def find_threshold_for_Pm(Pm, Rm, Rc, code_distances, p_list, runtime_budget,
                          threshold_method="adjacent", verbose=True, n_workers=1,
                          checkpoint_path=None):
    """
    하나의 Pm 값에 대해 threshold 추정
    
    Returns:
        threshold: 추정된 threshold (없으면 None)
        threshold_err: threshold 오차 (없으면 None)
        results: 시뮬레이션 결과 dict
    """
    print(f"\n{'='*60}")
    print(f"  Finding threshold for Pm = {Pm:.4f}")
    print(f"{'='*60}")

    results = run_p_sweep(Pm, Rm, Rc, code_distances, p_list, runtime_budget,
                          n_workers=n_workers, checkpoint_path=checkpoint_path)

    # Threshold 추정
    print(f"\n>>> Estimating threshold for Pm={Pm:.4f}...")
    threshold, threshold_err = estimate_threshold_from_data(
        results, code_distances, verbose=verbose, method=threshold_method
    )

    if threshold is not None:
        print(f"    ★ Pm={Pm:.4f} → p_th = {threshold:.6f} ({threshold*100:.3f}%)")
    else:
        print(f"    ✗ Pm={Pm:.4f} → threshold not found")

    return threshold, threshold_err, results


# ============== 저장/로드 ==============

def save_all_results(all_data, filename):
    """전체 결과를 JSON으로 저장"""
    serializable = {
        "Pm_list": all_data["Pm_list"],
        "thresholds": all_data["thresholds"],
        "threshold_errs": all_data["threshold_errs"],
        "params": all_data["params"],
        "per_Pm_results": {}
    }

    for Pm_key, results in all_data.get("per_Pm_results", {}).items():
        serializable["per_Pm_results"][str(Pm_key)] = {}
        for d, vals in results.items():
            serializable["per_Pm_results"][str(Pm_key)][str(d)] = {
                "p": [float(x) for x in vals["p"]],
                "pL": [float(x) for x in vals["pL"]],
                "pL_dev": [float(x) for x in vals["pL_dev"]]
            }

    with open(filename, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved all results to: {filename}")


def load_all_results(filename):
    """JSON에서 전체 결과 로드"""
    with open(filename, 'r') as f:
        data = json.load(f)

    # per_Pm_results의 key를 float으로 복원
    per_Pm = {}
    for Pm_key, res_dict in data.get("per_Pm_results", {}).items():
        per_Pm[float(Pm_key)] = {}
        for d_str, vals in res_dict.items():
            per_Pm[float(Pm_key)][int(d_str)] = vals

    data["per_Pm_results"] = per_Pm
    return data


# ============== 플롯 ==============

def plot_threshold_vs_Pm(Pm_list, thresholds, threshold_errs=None,
                         save_path="threshold_vs_Pm.pdf",
                         title="Threshold vs Measurement Error Probability (Re=0)"):
    """
    x축: Pm (measurement error probability)
    y축: p_th (threshold physical error rate)
    """
    # 유효한 데이터만 필터
    valid = [(Pm, th, err) for Pm, th, err in
             zip(Pm_list, thresholds,
                 threshold_errs if threshold_errs else [None]*len(Pm_list))
             if th is not None]

    if not valid:
        print("No valid threshold data to plot!")
        return

    Pm_valid = [v[0] for v in valid]
    th_valid = [v[1] for v in valid]
    err_valid = [v[2] for v in valid]

    fig, ax = plt.subplots(figsize=(8, 6))

    has_errors = all(e is not None for e in err_valid)
    if has_errors:
        ax.errorbar(Pm_valid, th_valid, yerr=err_valid,
                     fmt='o-', color='C0', markersize=7, linewidth=2,
                     capsize=4, capthick=1.5, markerfacecolor='C0',
                     markeredgecolor='C0', label='$p_{th}$ (Re=0)')
    else:
        ax.plot(Pm_valid, th_valid, 'o-', color='C0', markersize=7,
                linewidth=2, label='$p_{th}$ (Re=0)')

    ax.set_xlabel('Measurement error probability, $P_m$', fontsize=14)
    ax.set_ylabel('Threshold physical error rate, $p_{th}$', fontsize=14)
    ax.set_title(title, fontsize=13)

    ax.set_xlim(-0.001, max(Pm_valid) * 1.05 + 0.001)
    y_min = min(th_valid) * 0.9
    y_max = max(th_valid) * 1.1
    ax.set_ylim(y_min, y_max)

    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.legend(fontsize=12, loc='best')

    # 각 데이터 포인트에 값 표시
    for Pm, th in zip(Pm_valid, th_valid):
        ax.annotate(f'{th*100:.2f}%',
                    xy=(Pm, th), xytext=(5, 10),
                    textcoords='offset points', fontsize=8,
                    color='C0', ha='left')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {save_path}")
    plt.show()


def plot_individual_sweeps(all_data, save_dir):
    """각 Pm에 대한 개별 pL vs p 그래프도 저장 (디버깅용)"""
    per_Pm = all_data.get("per_Pm_results", {})
    if not per_Pm:
        return

    for Pm_val, results in sorted(per_Pm.items()):
        fig, ax = plt.subplots(figsize=(7, 5))
        code_distances = sorted(results.keys())
        colors = {3: 'C0', 5: 'C1', 7: 'C2', 9: 'C3', 11: 'C4', 13: 'C5', 15: 'C6'}

        for d in code_distances:
            color = colors.get(d, 'gray')
            p_arr = np.array(results[d]["p"])
            pL_arr = np.array(results[d]["pL"])
            valid = pL_arr > 0
            if np.any(valid):
                ax.plot(p_arr[valid], pL_arr[valid], 'o-', color=color,
                        markersize=5, linewidth=1.5, label=f'd={d}')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Physical error rate $p$', fontsize=12)
        ax.set_ylabel('Logical error rate $p_L$', fontsize=12)
        ax.set_title(f'$P_m = {Pm_val:.4f}$ (Re=0)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fname = os.path.join(save_dir, f"pL_vs_p_Pm{Pm_val:.4f}.pdf")
        plt.tight_layout()
        plt.savefig(fname, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved individual plot: {fname}")


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
    parser.add_argument('--threshold-method', default='adjacent',
                        choices=['adjacent', 'largest_pair', 'smallest_pair', 'all_pairs'],
                        help='Threshold estimation method')
    parser.add_argument('--plot-individual', action='store_true',
                        help='Also plot individual pL vs p for each Pm')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers (0 = all cores, 1 = sequential)')
    parser.add_argument('--fresh', action='store_true',
                        help='Delete existing checkpoints and start from scratch')
    args = parser.parse_args()
    args.parallel = resolve_parallel_workers(args.parallel)

    os.makedirs(args.data_dir, exist_ok=True)
    if args.fresh:
        clean_checkpoints(args.data_dir)
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
        Pm_list = np.linspace(0, 0.03, 7).tolist()  # [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
        code_distances = [5, 7, 9, 11]
        runtime_budget = (300, 45)
        p_sweep = np.logspace(-4, -1, 12).tolist()  # 물리적 에러율 sweep 범위

        print(f"    Pm values: {[f'{Pm:.4f}' for Pm in Pm_list]}")
        print(f"    Code distances: {code_distances}")
        print(f"    p sweep: {len(p_sweep)} points from {p_sweep[0]:.4e} to {p_sweep[-1]:.4e}")

        thresholds = []
        threshold_errs = []
        per_Pm_results = {}

        pm_tracker = ProgressTracker(len(Pm_list), "Pm values", print_every=1)
        for Pm in Pm_list:
            pm_tracker.begin_task()
            th, th_err, results = find_threshold_for_Pm(
                Pm, Rm, Rc, code_distances, p_sweep, runtime_budget,
                threshold_method=args.threshold_method,
                n_workers=args.parallel,
                checkpoint_path=os.path.join(args.data_dir, f"checkpoint_Pm{Pm:.4f}.json")
            )
            thresholds.append(th)
            threshold_errs.append(th_err)
            per_Pm_results[Pm] = results
            pm_tracker.end_task()
        pm_tracker.summary()

        # 결과 저장
        all_data = {
            "Pm_list": Pm_list,
            "thresholds": [float(t) if t is not None else None for t in thresholds],
            "threshold_errs": [float(e) if e is not None else None for e in threshold_errs],
            "params": {"Rm": Rm, "Rc": Rc, "code_distances": code_distances,
                       "mode": "quick", "threshold_method": args.threshold_method},
            "per_Pm_results": per_Pm_results,
        }
        save_all_results(all_data, os.path.join(args.data_dir, "results_quick.json"))

        # 결과 요약 출력
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

        # 그래프
        plot_threshold_vs_Pm(Pm_list, thresholds, threshold_errs,
                             save_path=args.output)

        if args.plot_individual:
            plot_individual_sweeps(all_data, args.data_dir)

        print(f"\n    End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    elif args.mode == 'full':
        print("=== Full mode: Threshold vs Pm (Re=0) ===")
        print(f"    Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ----- 설정 (더 촘촘하고, 더 큰 code distance, 더 많은 통계) -----
        Pm_list = np.linspace(0, 0.03, 13).tolist()  # 0, 0.0025, 0.005, ..., 0.03
        code_distances = [5, 7, 9, 11, 13]
        runtime_budget = (40000, 3600)
        p_sweep = np.logspace(-4, -1, 20).tolist()

        print(f"    Pm values: {[f'{Pm:.4f}' for Pm in Pm_list]}")
        print(f"    Code distances: {code_distances}")
        print(f"    p sweep: {len(p_sweep)} points from {p_sweep[0]:.4e} to {p_sweep[-1]:.4e}")

        thresholds = []
        threshold_errs = []
        per_Pm_results = {}

        pm_tracker = ProgressTracker(len(Pm_list), "Pm values", print_every=1)
        for Pm in Pm_list:
            pm_tracker.begin_task()
            th, th_err, results = find_threshold_for_Pm(
                Pm, Rm, Rc, code_distances, p_sweep, runtime_budget,
                threshold_method=args.threshold_method,
                n_workers=args.parallel,
                checkpoint_path=os.path.join(args.data_dir, f"checkpoint_Pm{Pm:.4f}_full.json")
            )
            thresholds.append(th)
            threshold_errs.append(th_err)
            per_Pm_results[Pm] = results
            pm_tracker.end_task()
        pm_tracker.summary()

        # 결과 저장
        all_data = {
            "Pm_list": Pm_list,
            "thresholds": [float(t) if t is not None else None for t in thresholds],
            "threshold_errs": [float(e) if e is not None else None for e in threshold_errs],
            "params": {"Rm": Rm, "Rc": Rc, "code_distances": code_distances,
                       "mode": "full", "threshold_method": args.threshold_method},
            "per_Pm_results": per_Pm_results,
        }
        save_all_results(all_data, os.path.join(args.data_dir, "results_full.json"))

        # 결과 요약 출력
        print(f"\n{'='*60}")
        print(f"  Summary: Threshold vs Pm (Re=0)")
        print(f"{'='*60}")
        print(f"  {'Pm':>8s}  {'p_th':>12s}  {'p_th (%)':>10s}  {'± err':>10s}")
        print(f"  {'-'*44}")
        for Pm, th, err in zip(Pm_list, thresholds, threshold_errs):
            if th is not None:
                err_str = f"{err*100:.3f}%" if err else "N/A"
                print(f"  {Pm:>8.4f}  {th:>12.6f}  {th*100:>9.3f}%  {err_str:>10s}")
            else:
                print(f"  {Pm:>8.4f}  {'N/A':>12s}  {'N/A':>10s}  {'N/A':>10s}")

        # 그래프
        plot_threshold_vs_Pm(Pm_list, thresholds, threshold_errs,
                             save_path=args.output)

        if args.plot_individual:
            plot_individual_sweeps(all_data, args.data_dir)

        print(f"\n    End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    elif args.mode == 'plot':
        print("=== Plot mode: loading saved results ===")

        # 저장된 결과 로드
        for fname in ["results_full.json", "results_quick.json"]:
            fpath = os.path.join(args.data_dir, fname)
            if os.path.exists(fpath):
                print(f"Loading: {fpath}")
                all_data = load_all_results(fpath)
                break
        else:
            print(f"No saved results found in {args.data_dir}/")
            sys.exit(1)

        Pm_list = all_data["Pm_list"]
        thresholds = all_data["thresholds"]
        threshold_errs = all_data["threshold_errs"]

        # threshold를 다시 계산할 수도 있음 (method 변경 시)
        if all_data.get("per_Pm_results"):
            code_distances = sorted(
                next(iter(all_data["per_Pm_results"].values())).keys()
            )
            print(f"  Recalculating thresholds with method='{args.threshold_method}'...")
            thresholds = []
            threshold_errs = []
            for Pm_val in Pm_list:
                results = all_data["per_Pm_results"].get(Pm_val, {})
                if results:
                    th, th_err = estimate_threshold_from_data(
                        results, code_distances, verbose=True,
                        method=args.threshold_method
                    )
                    thresholds.append(th)
                    threshold_errs.append(th_err)
                else:
                    thresholds.append(None)
                    threshold_errs.append(None)

        # 요약 출력
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

        # 그래프
        plot_threshold_vs_Pm(Pm_list, thresholds, threshold_errs,
                             save_path=args.output)

        if args.plot_individual:
            plot_individual_sweeps(all_data, args.data_dir)
