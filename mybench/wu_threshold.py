"""
Wu et al. Figure 3-a reproduction using ThresholdAnalyzer

이 스크립트는 threshold_analyzer.py의 ThresholdAnalyzer 클래스를 활용하여
Wu et al. 논문의 Figure 3-a를 재현합니다.

사용법:
    python wu_threshold.py --mode quick      # 빠른 테스트 (~10분)
    python wu_threshold.py --mode full       # 논문용 (~수 시간)
    python wu_threshold.py --mode plot       # 기존 데이터로 플롯만
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

def create_simulate_func(Re, noise_model_config=None):
    """
    특정 Re 값에 대한 simulate_func 생성
    
    Args:
        Re: erasure 비율 (0 ~ 1)
            p_pauli = p_total * (1 - Re)
            p_erasure = p_total * Re
        noise_model_config: 추가 noise model 설정 (dict)
    
    Returns:
        ThresholdAnalyzer에서 사용할 simulate_func
    """
    
    def simulate_func(p_total, d, runtime_budget, p_graph=None):
        min_error_cases, time_budget = runtime_budget
        noisy_measurements = d
        
        p_pauli = p_total * (1 - Re)
        p_erasure = p_total * Re
        
        # 기본 noise model configuration
        config = {"use_correlated_pauli": True, "use_correlated_erasure": True}
        if noise_model_config:
            config.update(noise_model_config)
        
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
            print(f"[ERROR] simulation failed for p={p_total}, d={d}, Re={Re}")
            return (0.5, 1.0)  # 실패 시 기본값
        
        # 결과 파싱
        # format: <p> <di> <nm> <shots> <failed> <pL> <dj> <pL_dev> <pe>
        full_result = stdout.strip(" \r\n").split("\n")[-1]
        lst = full_result.split(" ")
        pL = float(lst[5])
        pL_dev = float(lst[7])
        
        print(f"  [Re={Re}] d={d:2d}, p={p_total:.4e}: pL={pL:.4e} ± {pL_dev:.2e}")
        return (pL, pL_dev)
    
    return simulate_func


# ============== ThresholdAnalyzer 설정 ==============

def run_threshold_analysis(Re, code_distances, rough_code_distances,
                           rough_runtime_budget, runtime_budget,
                           save_image=None, verbose=True):
    """
    ThresholdAnalyzer를 사용하여 특정 Re에 대한 threshold 분석
    
    Returns:
        (threshold, threshold_error, collected_data)
    """
    print("\n" + "="*70)
    print(f" Threshold Analysis for Re = {Re}")
    print("="*70)
    _ta_start = time.time()
    
    simulate_func = create_simulate_func(Re)
    
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
    
    # Re에 따라 시작점 조정
    if Re > 0.9:
        # High erasure: threshold가 높음 (~4%)
        analyzer.rough_init_search_start_p = 0.15
    else:
        # Low erasure: threshold가 낮음 (~1%)
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
            print(f"\n>>> Re={Re}: Threshold (fitting) = {threshold_fit*100:.3f}% ± {threshold_fit_err*100:.3f}%")
            
            # 데이터 교차점으로도 계산 (이게 plot에 사용됨)
            results_for_crossing = extract_plot_data(analyzer.collected_data_list, distances)
            threshold_crossing, threshold_crossing_err = estimate_threshold_from_data(
                results_for_crossing, distances, verbose=True)
            if threshold_crossing is not None:
                print(f">>> Re={Re}: Threshold (crossing) = {threshold_crossing*100:.3f}%")
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


def run_p_sweep(Re, code_distances, p_list, runtime_budget):
    """
    고정된 p 값들에 대해 시뮬레이션 수행 (넓은 범위 데이터 수집용)
    
    Args:
        Re: erasure 비율
        code_distances: [5, 7, 9, 11, ...]
        p_list: 시뮬레이션할 p 값 목록
        runtime_budget: (min_error_cases, time_budget)
    
    Returns:
        {d: {"p": [...], "pL": [...], "pL_dev": [...]}}
    """
    print(f"\n>>> Running p-sweep for Re={Re}")
    print(f"    p values: {[f'{p:.4f}' for p in p_list]}")
    print(f"    code distances: {code_distances}")
    
    simulate_func = create_simulate_func(Re)
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


# merge_results is imported from utils.py


# ============== 플롯 함수 ==============

def plot_wu_fig3a(results_re0, results_re98, code_distances, 
                  threshold_re0=None, threshold_re98=None,
                  save_path="wu_fig3a.pdf", max_points_per_curve=15):
    """Wu et al. Figure 3-a 스타일 그래프
    
    threshold_re0, threshold_re98: 외부에서 전달받은 threshold 값을 그대로 사용
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
        
        # log scale에서 등간격 선택
        log_p = np.log10(p_valid)
        indices = np.linspace(0, len(p_valid) - 1, max_points, dtype=int)
        indices = np.unique(indices)  # 중복 제거
        
        return p_valid[indices].tolist(), pL_valid[indices].tolist()
    
    for d in code_distances:
        color = colors.get(d, 'gray')
        
        # Re = 0 (open circles, dashed lines)
        if d in results_re0 and len(results_re0[d]["p"]) > 0:
            p_plot, pL_plot = subsample_data(results_re0[d]["p"], results_re0[d]["pL"], max_points_per_curve)
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
        
        # Re = 0.98 (filled circles, solid lines)
        if d in results_re98 and len(results_re98[d]["p"]) > 0:
            p_plot, pL_plot = subsample_data(results_re98[d]["p"], results_re98[d]["pL"], max_points_per_curve)
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
    
    # Threshold 수직선 및 레이블
    if threshold_re0:
        ax.axvline(x=threshold_re0, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.annotate(f'$p_{{th}}^{{R_e=0}}$\n{threshold_re0*100:.2f}%', 
                    xy=(threshold_re0, 0.3), fontsize=9, color='blue',
                    ha='right', va='bottom')
    if threshold_re98:
        ax.axvline(x=threshold_re98, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.annotate(f'$p_{{th}}^{{R_e=0.98}}$\n{threshold_re98*100:.2f}%', 
                    xy=(threshold_re98, 0.3), fontsize=9, color='red',
                    ha='left', va='bottom')
    
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
    ax.text(0.02, 0.18, '$R_e = 0$\n(dashed, open)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.02, 0.08, '$R_e = 0.98$\n(solid, filled)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    
    # 제목
    title = "Circuit-level threshold with erasure conversion"
    if threshold_re0 and threshold_re98:
        title += f"\n$p_{{th}}(R_e=0)={threshold_re0*100:.2f}\\%$, $p_{{th}}(R_e=0.98)={threshold_re98*100:.2f}\\%$"
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


def fit_threshold_from_results(results, code_distances=None):
    """
    저장된 results 데이터에서 threshold를 fitting으로 계산
    
    Args:
        results: {d: {"p": [...], "pL": [...], "pL_dev": [...]}}
        code_distances: 사용할 code distances (None이면 모두 사용)
    
    Returns:
        (threshold, threshold_err) or (None, None) if fitting fails
    """
    from scipy.optimize import curve_fit
    
    if code_distances is None:
        code_distances = sorted(results.keys())
    
    # Quadratic approximation curve (ThresholdAnalyzer와 동일)
    def quadratic_approx_curve(p_d_pair, A, B, C, pc0, v0):
        y_list = []
        for p, d in p_d_pair:
            x = (p - pc0) * (d ** (1. / v0))
            y = A + B * x + C * (x ** 2)
            y_list.append(y)
        return y_list
    
    # 데이터 준비
    x_data = []
    y_data = []
    sigma = []
    
    for d in code_distances:
        if d not in results:
            continue
        for i, p in enumerate(results[d]["p"]):
            pL = results[d]["pL"][i]
            pL_dev = results[d]["pL_dev"][i] if i < len(results[d]["pL_dev"]) else 0.1
            if pL > 0 and pL < 1:  # 유효한 데이터만
                x_data.append((p, d))
                y_data.append(pL)
                sigma.append(max(pL_dev, 0.001))  # 최소 오차
    
    if len(x_data) < 5:
        print(f"[WARNING] Not enough data points for fitting: {len(x_data)}")
        return None, None
    
    try:
        # 초기 추측값
        p_values = [x[0] for x in x_data]
        guess_A = np.average(y_data)
        guess_pc0 = np.median(p_values)
        
        # Bounds 설정
        p_range = max(p_values) - min(p_values)
        lower_bounds = [min(y_data) * 0.1, -np.inf, -100, min(p_values) - p_range, 0.5]
        upper_bounds = [max(y_data) * 2, np.inf, 100, max(p_values) + p_range, 3]
        
        popt, pcov = curve_fit(
            quadratic_approx_curve, x_data, y_data,
            sigma=sigma, absolute_sigma=False,
            p0=[guess_A, 1, 0.1, guess_pc0, 1],
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
        
        threshold = popt[3]
        threshold_err = perr[3]
        
        print(f"[Fit] A={popt[0]:.4f}, B={popt[1]:.4f}, C={popt[2]:.4f}")
        print(f"      pc0={threshold:.6f} ± {threshold_err:.6f}, v0={popt[4]:.4f}")
        
        return threshold, threshold_err
    
    except Exception as e:
        print(f"[WARNING] Fitting failed: {e}")
        return None, None


# ============== 메인 ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Wu et al. Figure 3-a using ThresholdAnalyzer')
    parser.add_argument('--mode', choices=['quick', 'full', 'plot'], 
                        default='quick', help='Simulation mode')
    parser.add_argument('--output', default='wu_fig3a_threshold.pdf', help='Output file path')
    parser.add_argument('--data-dir', default='.', help='Directory for data files')
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    compile_code_if_necessary()
    
    if args.mode == 'quick':
        # 빠른 테스트 (~15-30분)
        print("=== Quick mode: Log-spaced p sweep + ThresholdAnalyzer ===")
        
        code_distances = [5, 7, 9, 11]
        rough_code_distances = [5, 7]
        rough_runtime_budget = (200, 30)    # 200 errors or 30 sec
        runtime_budget = (500, 60)          # 500 errors or 60 sec
        sweep_runtime_budget = (300, 45)    # sweep용
        
        # 논문처럼 10^-4 ~ 10^-1 로그 등간격 (quick: 12개 포인트)
        p_sweep_all = np.logspace(-4, -1, 12)
        
        # ========== Re = 0 (threshold ~0.5%) ==========
        # Re=0은 threshold가 낮으므로 10^-4 ~ 10^-2 범위
        # p_list_re0 = p_sweep_all[p_sweep_all <= 0.02]
        p_list_re0 = p_sweep_all
        print(f"\n>>> Re=0: p sweep from {p_list_re0[0]:.1e} to {p_list_re0[-1]:.1e} ({len(p_list_re0)} points)")
        results_re0_sweep = run_p_sweep(0, code_distances, p_list_re0.tolist(), sweep_runtime_budget)
        
        # ThresholdAnalyzer로 정밀 threshold 추정
        th_re0, th_re0_err, data_re0 = run_threshold_analysis(
            Re=0,
            code_distances=code_distances,
            rough_code_distances=rough_code_distances,
            rough_runtime_budget=rough_runtime_budget,
            runtime_budget=runtime_budget,
            save_image=os.path.join(args.data_dir, "threshold_re0.pdf")
        )
        results_re0_th = extract_plot_data(data_re0, code_distances)
        
        # 병합 및 저장
        results_re0 = merge_results(results_re0_sweep, results_re0_th)
        save_results(results_re0, th_re0, os.path.join(args.data_dir, "results_re0_th.json"))
        
        # ========== Re = 0.98 (threshold ~4%) ==========
        # Re=0.98은 threshold가 높으므로 10^-3 ~ 10^-1 범위
        # p_list_re98 = p_sweep_all[p_sweep_all >= 0.001]
        p_list_re98 = p_sweep_all
        print(f"\n>>> Re=0.98: p sweep from {p_list_re98[0]:.1e} to {p_list_re98[-1]:.1e} ({len(p_list_re98)} points)")
        results_re98_sweep = run_p_sweep(0.98, code_distances, p_list_re98.tolist(), sweep_runtime_budget)
        
        # ThresholdAnalyzer로 정밀 threshold 추정
        th_re98, th_re98_err, data_re98 = run_threshold_analysis(
            Re=0.98,
            code_distances=code_distances,
            rough_code_distances=rough_code_distances,
            rough_runtime_budget=rough_runtime_budget,
            runtime_budget=runtime_budget,
            save_image=os.path.join(args.data_dir, "threshold_re98.pdf")
        )
        results_re98_th = extract_plot_data(data_re98, code_distances)
        
        # 병합 및 저장
        results_re98 = merge_results(results_re98_sweep, results_re98_th)
        save_results(results_re98, th_re98, os.path.join(args.data_dir, "results_re98_th.json"))
        
        # Combined plot
        plot_wu_fig3a(results_re0, results_re98, code_distances,
                      threshold_re0=th_re0, threshold_re98=th_re98,
                      save_path=args.output)
        
    elif args.mode == 'full':
        # 논문용 (~수 시간)
        print("=== Full mode: Log-spaced p sweep + ThresholdAnalyzer (paper quality) ===")
        
        code_distances = [5, 7, 9, 11, 13, 15]
        rough_code_distances = [5, 9]
        rough_runtime_budget = (1000, 120)   # 1000 errors or 2 min
        runtime_budget = (5000, 600)         # 5000 errors or 10 min
        sweep_runtime_budget = (2000, 300)   # sweep용
        
        # 논문처럼 10^-4 ~ 10^-1 로그 등간격 (full: 20개 포인트)
        p_sweep_all = np.logspace(-4, -1, 20)
        
        # ========== Re = 0 (threshold ~0.5%) ==========
        p_list_re0 = p_sweep_all[p_sweep_all <= 0.02]
        print(f"\n>>> Re=0: p sweep from {p_list_re0[0]:.1e} to {p_list_re0[-1]:.1e} ({len(p_list_re0)} points)")
        results_re0_sweep = run_p_sweep(0, code_distances, p_list_re0.tolist(), sweep_runtime_budget)
        
        th_re0, th_re0_err, data_re0 = run_threshold_analysis(
            Re=0,
            code_distances=code_distances,
            rough_code_distances=rough_code_distances,
            rough_runtime_budget=rough_runtime_budget,
            runtime_budget=runtime_budget,
            save_image=os.path.join(args.data_dir, "threshold_re0_full.pdf")
        )
        results_re0_th = extract_plot_data(data_re0, code_distances)
        
        results_re0 = merge_results(results_re0_sweep, results_re0_th)
        save_results(results_re0, th_re0, os.path.join(args.data_dir, "results_re0_full.json"))
        
        # ========== Re = 0.98 (threshold ~4%) ==========
        p_list_re98 = p_sweep_all[p_sweep_all >= 0.001]
        print(f"\n>>> Re=0.98: p sweep from {p_list_re98[0]:.1e} to {p_list_re98[-1]:.1e} ({len(p_list_re98)} points)")
        results_re98_sweep = run_p_sweep(0.98, code_distances, p_list_re98.tolist(), sweep_runtime_budget)
        
        th_re98, th_re98_err, data_re98 = run_threshold_analysis(
            Re=0.98,
            code_distances=code_distances,
            rough_code_distances=rough_code_distances,
            rough_runtime_budget=rough_runtime_budget,
            runtime_budget=runtime_budget,
            save_image=os.path.join(args.data_dir, "threshold_re98_full.pdf")
        )
        results_re98_th = extract_plot_data(data_re98, code_distances)
        
        results_re98 = merge_results(results_re98_sweep, results_re98_th)
        save_results(results_re98, th_re98, os.path.join(args.data_dir, "results_re98_full.json"))
        
        # Combined plot
        plot_wu_fig3a(results_re0, results_re98, code_distances,
                      threshold_re0=th_re0, threshold_re98=th_re98,
                      save_path=args.output)
        
        # Threshold 결과 출력
        print("\n" + "="*70)
        print(" THRESHOLD SUMMARY (Full Mode)")
        print("="*70)
        print(f" Re = 0.00 (no erasure):   p_th = {th_re0*100:.3f}% ± {th_re0_err*100:.3f}%" if th_re0 else " Re = 0.00: FAILED")
        print(f" Re = 0.98 (98% erasure):  p_th = {th_re98*100:.3f}% ± {th_re98_err*100:.3f}%" if th_re98 else " Re = 0.98: FAILED")
        if th_re0 and th_re98:
            print(f"\n Improvement ratio: {th_re98/th_re0:.2f}x")
        print("="*70)
        
    elif args.mode == 'plot':
        # 기존 데이터로 플롯
        try:
            results_re0, th_re0 = load_results(os.path.join(args.data_dir, "results_re0_th.json"))
            results_re98, th_re98 = load_results(os.path.join(args.data_dir, "results_re98_th.json"))
        except FileNotFoundError:
            results_re0, th_re0 = load_results(os.path.join(args.data_dir, "results_re0_full.json"))
            results_re98, th_re98 = load_results(os.path.join(args.data_dir, "results_re98_full.json"))
        
        code_distances = sorted(results_re0.keys())
        
        # Threshold 계산 - crossing method 우선, 실패시 fitting
        th_re0_err, th_re98_err = None, None
        
        print("\n>>> Re=0: Calculating threshold from crossing points...")
        th_re0_cross, th_re0_cross_err = estimate_threshold_from_data(results_re0, code_distances, verbose=True)
        if th_re0_cross is not None:
            th_re0 = th_re0_cross
            th_re0_err = th_re0_cross_err
            print(f"    Threshold (crossing): {th_re0*100:.3f}%")
        else:
            print("    Crossing method failed, using fitting...")
            th_re0, th_re0_err = fit_threshold_from_results(results_re0, code_distances)
        
        print("\n>>> Re=0.98: Calculating threshold from crossing points...")
        th_re98_cross, th_re98_cross_err = estimate_threshold_from_data(results_re98, code_distances, verbose=True)
        if th_re98_cross is not None:
            th_re98 = th_re98_cross
            th_re98_err = th_re98_cross_err
            print(f"    Threshold (crossing): {th_re98*100:.3f}%")
        else:
            print("    Crossing method failed, using fitting...")
            th_re98, th_re98_err = fit_threshold_from_results(results_re98, code_distances)
        
        plot_wu_fig3a(results_re0, results_re98, code_distances,
                      threshold_re0=th_re0, threshold_re98=th_re98,
                      save_path=args.output)
        
        # Threshold 결과 출력
        print("\n" + "="*70)
        print(" THRESHOLD SUMMARY")
        print("="*70)
        if th_re0:
            err_str = f" ± {th_re0_err*100:.3f}%" if th_re0_err else ""
            print(f" Re = 0.00 (no erasure):   p_th = {th_re0*100:.3f}%{err_str}")
        else:
            print(" Re = 0.00: threshold fitting failed")
        if th_re98:
            err_str = f" ± {th_re98_err*100:.3f}%" if th_re98_err else ""
            print(f" Re = 0.98 (98% erasure):  p_th = {th_re98*100:.3f}%{err_str}")
        else:
            print(" Re = 0.98: threshold fitting failed")
        if th_re0 and th_re98:
            print(f"\n Improvement ratio: {th_re98/th_re0:.2f}x")
        print("="*70)
