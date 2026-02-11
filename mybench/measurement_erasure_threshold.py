"""
Measurement Erasure Threshold Analysis

이 스크립트는 measurement error에 erasure conversion을 적용했을 때의 threshold를 분석합니다.

모델:
  - Pm: measurement error rate (total)
  - Rm: erasure rate on measurement error (에러 발생 시 detect 확률)
  - Rc: erasure rate on correct measurement (정상 측정 시 결과 없는 확률)
  
  실제 적용:
  - measurement_error_rate = Pm * (1 - Rm)  # Pauli만
  - measurement_erasure_rate = Pm * Rm + (1 - Pm) * Rc  # 총 erasure

사용법:
    python measurement_erasure_threshold.py --mode quick
    python measurement_erasure_threshold.py --mode full
    python measurement_erasure_threshold.py --mode plot
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

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

from utils import find_crossing_point, estimate_threshold_from_data, merge_results, ProgressTracker, run_parallel_simulations, scaled_runtime_budget, resolve_parallel_workers


# ============== 시뮬레이션 함수 정의 ==============

def create_simulate_func(Pm, Rm, Rc=0.0):
    """
    Measurement erasure 모델에 대한 simulate_func 생성
    
    Args:
        Pm: total measurement error probability
        Rm: erasure rate on measurement error (0 ~ 1)
            = P(erasure | error)
        Rc: erasure rate on correct measurement (0 ~ 1)
            = P(erasure | no error), default 0
    
    Derived rates:
        measurement_error_rate = Pm * (1 - Rm)   # Pauli error only (hidden)
        measurement_error_rate_with_erasure = Pm * Rm  # erasure + Pauli error (detected)
        measurement_erasure_rate_no_error = (1 - Pm) * Rc  # erasure only, no error (false positive)
    
    Returns:
        ThresholdAnalyzer에서 사용할 simulate_func
    """
    
    # 실제 noise model 파라미터 계산
    measurement_error_rate = Pm * (1 - Rm)           # hidden Pauli error
    measurement_error_rate_with_erasure = Pm * Rm    # erasure + Pauli error
    measurement_erasure_rate_no_error = (1 - Pm) * Rc  # erasure only (false positive)
    
    print(f"[Model] Pm={Pm}, Rm={Rm}, Rc={Rc}")
    print(f"        → measurement_error_rate={measurement_error_rate:.6f} (hidden Pauli)")
    print(f"        → measurement_error_rate_with_erasure={measurement_error_rate_with_erasure:.6f} (erasure+error)")
    print(f"        → measurement_erasure_rate_no_error={measurement_erasure_rate_no_error:.6f} (erasure only)")
    
    def simulate_func(p, d, runtime_budget, p_graph=None):
        min_error_cases, time_budget = runtime_budget
        noisy_measurements = d
        
        # noise model configuration
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
        
        # p_graph 사용 (decoding graph 고정용)
        command = qecp_benchmark_simulate_func_command_vec(
            p, d, d, noisy_measurements, parameters,
            min_error_cases=min_error_cases,
            time_budget=time_budget,
            p_graph=p_graph
        )
        
        stdout, returncode = run_qecp_command_get_stdout(command)
        if returncode != 0:
            print(f"[ERROR] simulation failed for p={p}, d={d}")
            return (0.5, 1.0)  # 실패 시 기본값
        
        # 결과 파싱
        full_result = stdout.strip(" \r\n").split("\n")[-1]
        lst = full_result.split(" ")
        pL = float(lst[5])
        pL_dev = float(lst[7])
        
        print(f"  [Pm={Pm},Rm={Rm}] d={d:2d}, p={p:.4e}: pL={pL:.4e} ± {pL_dev:.2e}")
        return (pL, pL_dev)
    
    return simulate_func


def run_p_sweep(Pm, Rm, Rc, code_distances, p_list, runtime_budget, n_workers=1):
    """고정된 p 값들에 대해 시뮬레이션 수행"""
    print(f"\n>>> Running p-sweep for Pm={Pm}, Rm={Rm}, Rc={Rc}")
    print(f"    p values: {[f'{p:.4f}' for p in p_list]}")
    
    simulate_func = create_simulate_func(Pm, Rm, Rc)

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


# merge_results, find_crossing_point, estimate_threshold_from_data are imported from utils.py


# ============== 플롯 함수 ==============

def plot_measurement_erasure_comparison(results_rm0, results_rm98, code_distances, 
                                         Pm, threshold_rm0=None, threshold_rm98=None,
                                         save_path="measurement_erasure_threshold.pdf",
                                         max_points_per_curve=15):
    """Measurement erasure 비교 그래프
    
    threshold_rm0, threshold_rm98: 외부에서 전달받은 threshold 값을 그대로 사용
    """
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    colors = {3: 'C0', 5: 'C1', 7: 'C2', 9: 'C3', 11: 'C4', 13: 'C5', 15: 'C6'}
    
    def subsample_data(p_list, pL_list, max_points):
        """등간격으로 데이터 포인트 선택"""
        p_arr = np.array(p_list)
        pL_arr = np.array(pL_list)
        
        valid = pL_arr > 0
        p_valid = p_arr[valid]
        pL_valid = pL_arr[valid]
        
        if len(p_valid) <= max_points:
            return p_valid.tolist(), pL_valid.tolist()
        
        indices = np.linspace(0, len(p_valid) - 1, max_points, dtype=int)
        indices = np.unique(indices)
        
        return p_valid[indices].tolist(), pL_valid[indices].tolist()
    
    for d in code_distances:
        color = colors.get(d, 'gray')
        
        # Rm = 0 (no erasure detection, open circles, dashed)
        if d in results_rm0 and len(results_rm0[d]["p"]) > 0:
            p_plot, pL_plot = subsample_data(results_rm0[d]["p"], results_rm0[d]["pL"], max_points_per_curve)
            ax.errorbar(
                p_plot,
                pL_plot,
                fmt='o--',
                color=color,
                markerfacecolor='white',
                markeredgecolor=color,
                markersize=6,
                linewidth=1.5,
            )
        
        # Rm = 0.98 (high erasure detection, filled circles, solid)
        if d in results_rm98 and len(results_rm98[d]["p"]) > 0:
            p_plot, pL_plot = subsample_data(results_rm98[d]["p"], results_rm98[d]["pL"], max_points_per_curve)
            ax.errorbar(
                p_plot,
                pL_plot,
                fmt='o-',
                color=color,
                markerfacecolor=color,
                markersize=6,
                linewidth=1.5,
                label=f'd = {d}'
            )
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical error probability, $p$', fontsize=14)
    ax.set_ylabel('Logical error rate, $p_L$', fontsize=14)
    ax.set_xlim(1e-4, 2e-1)
    ax.set_ylim(1e-6, 1)
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Threshold 수직선
    if threshold_rm0 is not None:
        ax.axvline(x=threshold_rm0, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        ax.text(threshold_rm0*1.1, 1e-5, f'$p_{{th}}={threshold_rm0*100:.2f}\\%$\n(Rm=0)', 
                fontsize=9, color='gray')
    if threshold_rm98 is not None:
        ax.axvline(x=threshold_rm98, color='blue', linestyle=':', alpha=0.7, linewidth=2)
        ax.text(threshold_rm98*0.5, 1e-5, f'$p_{{th}}={threshold_rm98*100:.2f}\\%$\n(Rm=0.98)', 
                fontsize=9, color='blue')
    
    # 범례 설명
    ax.text(0.02, 0.18, f'$R_m = 0$\n(dashed, open)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.02, 0.08, f'$R_m = 0.98$\n(solid, filled)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    
    title = f"Measurement erasure conversion (Pm={Pm})"
    if threshold_rm0 is not None and threshold_rm98 is not None:
        title += f"\n$p_{{th}}(R_m=0)={threshold_rm0*100:.2f}\\%$, $p_{{th}}(R_m=0.98)={threshold_rm98*100:.2f}\\%$"
    ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {save_path}")
    plt.show()


def save_results(results, params, filename):
    """결과를 JSON으로 저장"""
    data = {
        "params": params,
        "results": {}
    }
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Measurement Erasure Threshold Analysis')
    parser.add_argument('--mode', choices=['quick', 'full', 'plot'], 
                        default='quick', help='Simulation mode')
    parser.add_argument('--Pm', type=float, default=0.02, help='Measurement error probability')
    parser.add_argument('--output', default='measurement_erasure_threshold.pdf', help='Output file path')
    parser.add_argument('--data-dir', default='results_measurement_erasure', help='Directory for data files')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel workers (0 = all cores, 1 = sequential)')
    args = parser.parse_args()
    args.parallel = resolve_parallel_workers(args.parallel)
    
    os.makedirs(args.data_dir, exist_ok=True)
    if not os.path.isabs(args.output):
        args.output = os.path.join(args.data_dir, args.output)
    compile_code_if_necessary()
    
    Pm = args.Pm
    Rc = 0.0  # 정상 측정에서는 erasure 없음
    
    if args.mode == 'quick':
        print(f"=== Quick mode: Measurement Erasure Comparison (Pm={Pm}) ===")
        
        code_distances = [5, 7, 9, 11]
        runtime_budget = (300, 45)
        
        # 10^-4 ~ 10^-1 로그 등간격
        p_sweep_all = np.logspace(-4, -1, 12)
        
        # ========== Rm = 0 (no erasure detection) ==========
        print("\n" + "="*60)
        print(f" Case 1: Rm = 0 (no measurement erasure detection)")
        print("="*60)
        results_rm0 = run_p_sweep(Pm, Rm=0, Rc=Rc, 
                                   code_distances=code_distances, 
                                   p_list=p_sweep_all.tolist(), 
                                   runtime_budget=runtime_budget,
                                   n_workers=args.parallel)
        save_results(results_rm0, {"Pm": Pm, "Rm": 0, "Rc": Rc}, 
                     os.path.join(args.data_dir, "results_rm0.json"))
        
        # ========== Rm = 0.98 (high erasure detection) ==========
        print("\n" + "="*60)
        print(f" Case 2: Rm = 0.98 (high measurement erasure detection)")
        print("="*60)
        results_rm98 = run_p_sweep(Pm, Rm=0.98, Rc=Rc,
                                    code_distances=code_distances,
                                    p_list=p_sweep_all.tolist(),
                                    runtime_budget=runtime_budget,
                                    n_workers=args.parallel)
        save_results(results_rm98, {"Pm": Pm, "Rm": 0.98, "Rc": Rc}, 
                     os.path.join(args.data_dir, "results_rm98.json"))
        
        # Combined plot
        plot_measurement_erasure_comparison(results_rm0, results_rm98, code_distances,
                                             Pm=Pm, save_path=args.output)
        
    elif args.mode == 'full':
        print(f"=== Full mode: Measurement Erasure Comparison (Pm={Pm}) ===")
        
        code_distances = [5, 7, 9, 11, 13]
        runtime_budget = (1000, 180)
        
        p_sweep_all = np.logspace(-4, -1, 20)
        
        # Rm = 0
        results_rm0 = run_p_sweep(Pm, Rm=0, Rc=Rc,
                                   code_distances=code_distances,
                                   p_list=p_sweep_all.tolist(),
                                   runtime_budget=runtime_budget,
                                   n_workers=args.parallel)
        save_results(results_rm0, {"Pm": Pm, "Rm": 0, "Rc": Rc},
                     os.path.join(args.data_dir, "results_rm0_full.json"))
        
        # Rm = 0.98
        results_rm98 = run_p_sweep(Pm, Rm=0.98, Rc=Rc,
                                    code_distances=code_distances,
                                    p_list=p_sweep_all.tolist(),
                                    runtime_budget=runtime_budget,
                                    n_workers=args.parallel)
        save_results(results_rm98, {"Pm": Pm, "Rm": 0.98, "Rc": Rc},
                     os.path.join(args.data_dir, "results_rm98_full.json"))
        
        plot_measurement_erasure_comparison(results_rm0, results_rm98, code_distances,
                                             Pm=Pm, save_path=args.output)
        
    elif args.mode == 'plot':
        try:
            results_rm0, _ = load_results(os.path.join(args.data_dir, "results_rm0.json"))
            results_rm98, params = load_results(os.path.join(args.data_dir, "results_rm98.json"))
        except FileNotFoundError:
            results_rm0, _ = load_results(os.path.join(args.data_dir, "results_rm0_full.json"))
            results_rm98, params = load_results(os.path.join(args.data_dir, "results_rm98_full.json"))
        
        Pm = params.get("Pm", 0.02)
        code_distances = sorted(results_rm0.keys())
        
        # Threshold 계산 - crossing method 우선
        print("\n>>> Rm=0: Calculating threshold from crossing points...")
        th_rm0, th_rm0_err = estimate_threshold_from_data(results_rm0, code_distances, verbose=True)
        if th_rm0 is not None:
            print(f"    Threshold (crossing): {th_rm0*100:.3f}%")
        
        print("\n>>> Rm=0.98: Calculating threshold from crossing points...")
        th_rm98, th_rm98_err = estimate_threshold_from_data(results_rm98, code_distances, verbose=True)
        if th_rm98 is not None:
            print(f"    Threshold (crossing): {th_rm98*100:.3f}%")
        
        plot_measurement_erasure_comparison(results_rm0, results_rm98, code_distances,
                                             Pm=Pm, threshold_rm0=th_rm0, threshold_rm98=th_rm98,
                                             save_path=args.output)
