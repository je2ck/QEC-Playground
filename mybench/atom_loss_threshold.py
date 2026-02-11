"""
Atom Loss Threshold Analysis

이 스크립트는 survival probability가 threshold에 미치는 영향을 분석합니다.
- pl = 0: loss error 없음

사용법:
    python atom_loss_threshold.py --mode quick                    # Re=0 (기본)
    python atom_loss_threshold.py --mode quick --Re 0.98          # Re=0.98
    python atom_loss_threshold.py --mode plot --Re 0              # Re=0 플롯
    python atom_loss_threshold.py --mode plot --Re 0.98           # Re=0.98 플롯
"""

import os
import sys
import json
import time
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

from utils import find_crossing_point, estimate_threshold_from_data, merge_results, ProgressTracker, format_duration


# ============== 시뮬레이션 함수 정의 ==============

def create_simulate_func(pl, Re=0):
    """
    특정 measurement error rate와 erasure ratio에 대한 simulate_func 생성
    
    Args:
        pm: measurement error rate (0 ~ 1)
        Re: erasure ratio - gate error 중 erasure로 변환되는 비율 (0 ~ 1)
    
    Returns:
        ThresholdAnalyzer에서 사용할 simulate_func
    """
    
    def simulate_func(p_total, d, runtime_budget, p_graph=None):
        min_error_cases, time_budget = runtime_budget
        noisy_measurements = d
        
        # p_total을 Pauli와 erasure로 분리
        p_pauli = p_total * (1 - Re)
        p_erasure = p_total * Re
        
        # noise model configuration with measurement error
        config = {
            "use_correlated_pauli": True,
            "use_correlated_erasure": True,
            "ancilla_loss_probability": pl,
        }
        
        parameters = [
            "--code-type", "rotated-planar-code",
            "--noise-model-builder", "only-gate-error-circuit-level",
            "--noise-model-configuration", json.dumps(config),
            "--decoder", "union-find",
            "--decoder-config", '{"pcmg":true}',
        ]
        
        # erasure가 있으면 pes 추가
        if p_erasure > 0:
            parameters += ["--pes", f"[{p_erasure:.10e}]"]
        
        # p_graph 사용 (decoding graph 고정용)
        command = qecp_benchmark_simulate_func_command_vec(
            p_pauli, d, d, noisy_measurements, parameters,
            min_error_cases=min_error_cases,
            time_budget=time_budget,
            p_graph=p_graph
        )
        
        stdout, returncode = run_qecp_command_get_stdout(command)
        if returncode != 0:
            print(f"[ERROR] simulation failed for p={p_total}, d={d}, pl={pl}, Re={Re}")
            return (0.5, 1.0)  # 실패 시 기본값
        
        # 결과 파싱
        # format: <p> <di> <nm> <shots> <failed> <pL> <dj> <pL_dev> <pe>
        full_result = stdout.strip(" \r\n").split("\n")[-1]
        lst = full_result.split(" ")
        pL = float(lst[5])
        pL_dev = float(lst[7])
        
        print(f"  [pl={pl}, Re={Re}] d={d:2d}, p={p_total:.4e}: pL={pL:.4e} ± {pL_dev:.2e}")
        return (pL, pL_dev)
    
    return simulate_func


# ============== ThresholdAnalyzer 설정 ==============

def run_threshold_analysis(pl, Re, code_distances, rough_code_distances,
                           rough_runtime_budget, runtime_budget,
                           save_image=None, verbose=True):
    """
    ThresholdAnalyzer를 사용하여 특정 pl과 Re에 대한 threshold 분석
    
    Returns:
        (threshold, threshold_error, collected_data)
    """
    print("\n" + "="*70)
    print(f" Threshold Analysis for pl = {pl}, Re = {Re}")
    print("="*70)
    _ta_start = time.time()
    
    simulate_func = create_simulate_func(pl, Re)
    
    analyzer = ThresholdAnalyzer(
        code_distances=code_distances,
        simulate_func=simulate_func,
        default_rough_runtime_budget=rough_runtime_budget,
        default_runtime_budget=runtime_budget
    )
    
    # 설정 조정
    analyzer.rough_code_distances = rough_code_distances
    analyzer.rough_runtime_budgets = [rough_runtime_budget] * len(rough_code_distances)
    analyzer.verbose = verbose
    
    # 시작점 조정 (gate error threshold ~0.5-1%)
    analyzer.rough_init_search_start_p = 0.05
    
    # Threshold 분석 실행
    try:
        analyzer.estimate(save_image=save_image)
        # estimate()가 return 없이 끝나므로 직접 fit_results 호출
        if analyzer.collected_data_list:
            distances, p_list, collected_data = analyzer.collected_data_list[-1]
            popt, perr = analyzer.fit_results(collected_data, p_list, distances)
            threshold_fit = popt[3]  # ThresholdAnalyzer 피팅 결과 (pc0)
            threshold_fit_err = perr[3]
            print(f"\n>>> pl={pl}: Threshold (fitting) = {threshold_fit*100:.3f}% ± {threshold_fit_err*100:.3f}%")
            
            # 데이터 교차점으로도 계산 (이게 plot에 사용됨)
            results_for_crossing = extract_plot_data(analyzer.collected_data_list, distances)
            threshold_crossing, threshold_crossing_err = estimate_threshold_from_data(
                results_for_crossing, distances, verbose=True)
            if threshold_crossing is not None:
                print(f">>> pl={pl}: Threshold (crossing) = {threshold_crossing*100:.3f}%")
                threshold = threshold_crossing
                threshold_err = threshold_crossing_err if threshold_crossing_err else threshold_fit_err
            else:
                threshold = threshold_fit
                threshold_err = threshold_fit_err
        else:
            print("[ERROR] No data collected")
            threshold, threshold_err = None, None
    except Exception as e:
        print(f"[ERROR] Threshold estimation failed: {e}")
        import traceback
        traceback.print_exc()
        threshold, threshold_err = None, None
    
    print(f"\n  \u23f1  ThresholdAnalyzer finished in {format_duration(time.time() - _ta_start)}")
    return threshold, threshold_err, analyzer.collected_data_list


def extract_plot_data(collected_data_list, code_distances):
    """
    ThresholdAnalyzer의 collected_data_list에서 플롯용 데이터 추출
    
    Returns:
        {d: {"p": [...], "pL": [...], "pL_dev": [...]}}
    """
    results = {d: {"p": [], "pL": [], "pL_dev": []} for d in code_distances}
    
    for (distances, p_list, collected_data) in collected_data_list:
        for i, d in enumerate(distances):
            if d in results:
                for j, p in enumerate(p_list):
                    pL, pL_dev = collected_data[i][j]
                    results[d]["p"].append(p)
                    results[d]["pL"].append(pL)
                    results[d]["pL_dev"].append(pL_dev)
    
    # 정렬 (p 기준)
    for d in results:
        if len(results[d]["p"]) > 0:
            sorted_indices = np.argsort(results[d]["p"])
            results[d]["p"] = [results[d]["p"][i] for i in sorted_indices]
            results[d]["pL"] = [results[d]["pL"][i] for i in sorted_indices]
            results[d]["pL_dev"] = [results[d]["pL_dev"][i] for i in sorted_indices]
    
    return results


def run_p_sweep(pl, Re, code_distances, p_list, runtime_budget):
    """
    고정된 p 값들에 대해 시뮬레이션 수행 (넓은 범위 데이터 수집용)
    
    Args:
        pl: ancilla loss probability
        Re: erasure ratio
        code_distances: [5, 7, 9, 11, ...]
        p_list: 시뮬레이션할 p 값 목록
        runtime_budget: (min_error_cases, time_budget)
    
    Returns:
        {d: {"p": [...], "pL": [...], "pL_dev": [...]}}
    """
    print(f"\n>>> Running p-sweep for pl={pl}, Re={Re}")
    print(f"    p values: {[f'{p:.4f}' for p in p_list]}")
    print(f"    code distances: {code_distances}")
    
    simulate_func = create_simulate_func(pl, Re)
    results = {d: {"p": [], "pL": [], "pL_dev": []} for d in code_distances}

    total_sims = len(p_list) * len(code_distances)
    tracker = ProgressTracker(total_sims, "simulations", print_every=len(code_distances))

    for p in p_list:
        print(f"\n--- p = {p:.4e} ---")
        for d in code_distances:
            tracker.begin_task()
            pL, pL_dev = simulate_func(p, d, runtime_budget, p_graph=p)
            results[d]["p"].append(p)
            results[d]["pL"].append(pL)
            results[d]["pL_dev"].append(pL_dev)
            tracker.end_task()

    tracker.summary()
    return results


# merge_results, find_crossing_point, estimate_threshold_from_data are imported from utils.py


# ============== 플롯 함수 ==============

def plot_loss_error_comparison(results_pl0, results_pl002, code_distances, 
                                       threshold_pl0=None, threshold_pl002=None,
                                       Re=0,
                                       save_path="loss_error_threshold.pdf",
                                       max_points_per_curve=15):
    """Loss error 비교 그래프
    
    threshold_pl0, threshold_pl002: 외부에서 전달받은 threshold 값을 그대로 사용
    Re: erasure ratio (제목에 표시용)
    """
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    colors = {3: 'C0', 5: 'C1', 7: 'C2', 9: 'C3', 11: 'C4', 13: 'C5', 15: 'C6'}
    
    def subsample_data(p_list, pL_list, max_points):
        """등간격으로 데이터 포인트 선택 (log scale 기준)"""
        p_arr = np.array(p_list)
        pL_arr = np.array(pL_list)
        
        # 유효한 데이터만 (pL > 0)
        valid = pL_arr > 0
        p_valid = p_arr[valid]
        pL_valid = pL_arr[valid]
        
        if len(p_valid) <= max_points:
            return p_valid.tolist(), pL_valid.tolist()
        
        # 인덱스 등간격 선택
        indices = np.linspace(0, len(p_valid) - 1, max_points, dtype=int)
        indices = np.unique(indices)
        
        return p_valid[indices].tolist(), pL_valid[indices].tolist()
    
    for d in code_distances:
        color = colors.get(d, 'gray')
        
        # pl = 0 (open circles, dashed lines)
        if d in results_pl0 and len(results_pl0[d]["p"]) > 0:
            p_plot, pL_plot = subsample_data(results_pl0[d]["p"], results_pl0[d]["pL"], max_points_per_curve)
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
        
        # pl = 0.02 (filled circles, solid lines)
        if d in results_pl002 and len(results_pl002[d]["p"]) > 0:
            p_plot, pL_plot = subsample_data(results_pl002[d]["p"], results_pl002[d]["pL"], max_points_per_curve)
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
    
    # Threshold 수직선
    if threshold_pl0:
        ax.axvline(x=threshold_pl0, color='gray', linestyle=':', alpha=0.7)
        ax.text(threshold_pl0*1.1, 1e-5, f'{threshold_pl0*100:.2f}%', fontsize=9, color='gray')
    if threshold_pl002:
        ax.axvline(x=threshold_pl002, color='blue', linestyle=':', alpha=0.7)
        ax.text(threshold_pl002*0.7, 1e-5, f'{threshold_pl002*100:.2f}%', fontsize=9, color='blue')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical error probability, $p$', fontsize=14)
    ax.set_ylabel('Logical error rate, $p_L$', fontsize=14)
    ax.set_xlim(1e-4, 2e-1)
    ax.set_ylim(1e-6, 1)
    
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # 범례 설명
    ax.text(0.02, 0.18, '$p_m = 0$\n(dashed, open)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.02, 0.08, '$p_m = 0.02$\n(solid, filled)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    
    # 제목
    title = f"Circuit-level threshold with measurement error (Re={Re})"
    if threshold_pl0 and threshold_pl002:
        title += f"\n$p_{{th}}(p_m=0)={threshold_pl0*100:.2f}\\%$, $p_{{th}}(p_m=0.02)={threshold_pl002*100:.2f}\\%$"
    ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {save_path}")
    plt.show()


def save_results(results, threshold, filename):
    """결과를 JSON으로 저장"""
    data = {
        "threshold": threshold,
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
    
    threshold = data.get("threshold")
    results = {}
    for d_str, vals in data["results"].items():
        results[int(d_str)] = vals
    
    return results, threshold


# ============== 메인 ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Measurement Error Threshold Analysis')
    parser.add_argument('--mode', choices=['quick', 'full', 'plot'], 
                        default='quick', help='Simulation mode')
    parser.add_argument('--Re', type=float, default=0, 
                        help='Erasure ratio (0 or 0.98). Gate errors are converted to erasures with this ratio.')
    parser.add_argument('--output', default=None, help='Output file path (default: auto-generated based on Re)')
    parser.add_argument('--data-dir', default=None, help='Directory for data files (default: results_measurement_error_ReXX)')
    args = parser.parse_args()
    
    # Re 값에 따라 디렉토리와 파일명 자동 설정
    Re_str = f"Re{int(args.Re*100):02d}"  # Re=0 -> "Re00", Re=0.98 -> "Re98"
    if args.data_dir is None:
        args.data_dir = f'results_measurement_error_{Re_str}'
    if args.output is None:
        args.output = f'measurement_error_threshold_{Re_str}.pdf'
    
    os.makedirs(args.data_dir, exist_ok=True)
    # output 파일도 data-dir에 저장
    if not os.path.isabs(args.output):
        args.output = os.path.join(args.data_dir, args.output)
    compile_code_if_necessary()
    
    print(f"\n>>> Configuration: Re={args.Re}, data_dir={args.data_dir}")
    
    if args.mode == 'quick':
        # 빠른 테스트 (~15-30분)
        print(f"=== Quick mode: Log-spaced p sweep + ThresholdAnalyzer (Re={args.Re}) ===")
        
        code_distances = [5, 7, 9, 11]
        rough_code_distances = [5, 7]
        rough_runtime_budget = (200, 30)    # 200 errors or 30 sec
        runtime_budget = (500, 60)          # 500 errors or 60 sec
        sweep_runtime_budget = (300, 45)    # sweep용
        
        # 논문처럼 10^-4 ~ 10^-1 로그 등간격 (quick: 12개 포인트)
        p_sweep_all = np.logspace(-4, -1, 12)
        
        # ========== pl = 0.00076 (0.076% measurement loss) ==========
        # p_list_pl0 = p_sweep_all[p_sweep_all <= 0.02]
        p_list_pl0 = p_sweep_all
        print(f"\n>>> pl=0.00076, Re={args.Re}: p sweep from {p_list_pl0[0]:.1e} to {p_list_pl0[-1]:.1e} ({len(p_list_pl0)} points)")
        results_pl0_sweep = run_p_sweep(0.00076, args.Re, code_distances, p_list_pl0.tolist(), sweep_runtime_budget)
        
        # ThresholdAnalyzer로 정밀 threshold 추정
        th_pl0, th_pl0_err, data_pl0 = run_threshold_analysis(
            pl=0.00076,
            Re=args.Re,
            code_distances=code_distances,
            rough_code_distances=rough_code_distances,
            rough_runtime_budget=rough_runtime_budget,
            runtime_budget=runtime_budget,
            save_image=os.path.join(args.data_dir, "threshold_pl0.pdf")
        )
        results_pl0_th = extract_plot_data(data_pl0, code_distances)
        
        # 병합 및 저장
        results_pl0 = merge_results(results_pl0_sweep, results_pl0_th)
        save_results(results_pl0, th_pl0, os.path.join(args.data_dir, "results_pl0.json"))
        
        # ========== pl = 0.0019 (0.19% measurement loss) ==========
        # p_list_pm002 = p_sweep_all[p_sweep_all <= 0.02]
        p_list_pl002 = p_sweep_all
        print(f"\n>>> pl=0.0019, Re={args.Re}: p sweep from {p_list_pl002[0]:.1e} to {p_list_pl002[-1]:.1e} ({len(p_list_pl002)} points)")
        results_pl002_sweep = run_p_sweep(0.0019, args.Re, code_distances, p_list_pl002.tolist(), sweep_runtime_budget)
        
        # ThresholdAnalyzer로 정밀 threshold 추정
        th_pl002, th_pl002_err, data_pl002 = run_threshold_analysis(
            pl=0.0019,
            Re=args.Re,
            code_distances=code_distances,
            rough_code_distances=rough_code_distances,
            rough_runtime_budget=rough_runtime_budget,
            runtime_budget=runtime_budget,
            save_image=os.path.join(args.data_dir, "threshold_pl002.pdf")
        )
        results_pl002_th = extract_plot_data(data_pl002, code_distances)
        
        # 병합 및 저장
        results_pl002 = merge_results(results_pl002_sweep, results_pl002_th)
        save_results(results_pl002, th_pl002, os.path.join(args.data_dir, "results_pl002.json"))
        
        # Combined plot
        plot_loss_error_comparison(results_pl0, results_pl002, code_distances,
                                           threshold_pl0=th_pl0, threshold_pl002=th_pl002,
                                           Re=args.Re,
                                           save_path=args.output)
        
    elif args.mode == 'full':
        # 논문용 (~수 시간)
        print(f"=== Full mode: Log-spaced p sweep + ThresholdAnalyzer (Re={args.Re}, paper quality) ===")
        
        code_distances = [5, 7, 9, 11, 13, 15]
        rough_code_distances = [5, 9]
        rough_runtime_budget = (1000, 120)   # 1000 errors or 2 min
        runtime_budget = (5000, 600)         # 5000 errors or 10 min
        sweep_runtime_budget = (2000, 300)   # sweep용
        
        # 논문처럼 10^-4 ~ 10^-1 로그 등간격 (full: 20개 포인트)
        p_sweep_all = np.logspace(-4, -1, 20)
        
        # ========== pl = 0.00076 ==========
        p_list_pl0 = p_sweep_all[p_sweep_all <= 0.03]
        print(f"\n>>> pl=0.00076, Re={args.Re}: p sweep from {p_list_pl0[0]:.1e} to {p_list_pl0[-1]:.1e} ({len(p_list_pl0)} points)")
        results_pl0_sweep = run_p_sweep(0.00076, args.Re, code_distances, p_list_pl0.tolist(), sweep_runtime_budget)
        
        th_pl0, th_pl0_err, data_pl0 = run_threshold_analysis(
            pl=0.00076,
            Re=args.Re,
            code_distances=code_distances,
            rough_code_distances=rough_code_distances,
            rough_runtime_budget=rough_runtime_budget,
            runtime_budget=runtime_budget,
            save_image=os.path.join(args.data_dir, "threshold_pl0_full.pdf")
        )
        results_pl0_th = extract_plot_data(data_pl0, code_distances)
        
        results_pl0 = merge_results(results_pl0_sweep, results_pl0_th)
        save_results(results_pl0, th_pl0, os.path.join(args.data_dir, "results_pl0_full.json"))
        
        # ========== pl = 0.0019 ==========
        p_list_pl002 = p_sweep_all[p_sweep_all <= 0.03]
        print(f"\n>>> pl=0.0019, Re={args.Re}: p sweep from {p_list_pl002[0]:.1e} to {p_list_pl002[-1]:.1e} ({len(p_list_pl002)} points)")
        results_pl002_sweep = run_p_sweep(0.0019, args.Re, code_distances, p_list_pl002.tolist(), sweep_runtime_budget)
        
        th_pl002, th_pl002_err, data_pl002 = run_threshold_analysis(
            pl=0.0019,
            Re=args.Re,
            code_distances=code_distances,
            rough_code_distances=rough_code_distances,
            rough_runtime_budget=rough_runtime_budget,
            runtime_budget=runtime_budget,
            save_image=os.path.join(args.data_dir, "threshold_pl002_full.pdf")
        )
        results_pl002_th = extract_plot_data(data_pl002, code_distances)
        
        results_pl002 = merge_results(results_pl002_sweep, results_pl002_th)
        save_results(results_pl002, th_pl002, os.path.join(args.data_dir, "results_pl002_full.json"))
        
        # Combined plot
        plot_loss_error_comparison(results_pl0, results_pl002, code_distances,
                                           threshold_pl0=th_pl0, threshold_pl002=th_pl002,
                                           Re=args.Re,
                                           save_path=args.output)
        
    elif args.mode == 'plot':
        # 기존 데이터로 플롯
        print(f"\n>>> Loading data from {args.data_dir} (Re={args.Re})")
        try:
            results_pl0, th_pl0 = load_results(os.path.join(args.data_dir, "results_pl0.json"))
            results_pl002, th_pl002 = load_results(os.path.join(args.data_dir, "results_pl002.json"))
        except FileNotFoundError:
            results_pl0, th_pl0 = load_results(os.path.join(args.data_dir, "results_pl0_full.json"))
            results_pl002, th_pl002 = load_results(os.path.join(args.data_dir, "results_pl002_full.json"))
        
        code_distances = sorted(results_pl0.keys())
        
        # Threshold 계산 - crossing method 우선
        th_pl0_err, th_pl002_err = None, None
        
        print(f"\n>>> pl=0.00076, Re={args.Re}: Calculating threshold from crossing points...")
        th_pl0_cross, th_pl0_cross_err = estimate_threshold_from_data(results_pl0, code_distances, verbose=True)
        if th_pl0_cross is not None:
            th_pl0 = th_pl0_cross
            th_pl0_err = th_pl0_cross_err
            print(f"    Threshold (crossing): {th_pl0*100:.3f}%")
        else:
            print("    Crossing method failed, using saved threshold")
        
        print(f"\n>>> pl=0.0019, Re={args.Re}: Calculating threshold from crossing points...")
        th_pl002_cross, th_pl002_cross_err = estimate_threshold_from_data(results_pl002, code_distances, verbose=True)
        if th_pl002_cross is not None:
            th_pl002 = th_pl002_cross
            th_pl002_err = th_pl002_cross_err
            print(f"    Threshold (crossing): {th_pl002*100:.3f}%")
        else:
            print("    Crossing method failed, using saved threshold")
        
        plot_loss_error_comparison(results_pl0, results_pl002, code_distances,
                                           threshold_pl0=th_pl0, threshold_pl002=th_pl002,
                                           Re=args.Re,
                                           save_path=args.output)
