import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import json

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
    compile_code_if_necessary
)

# ============== 설정 ==============

# Code distances (논문: d=3,5,7,9,11,13,15)
code_distances = [3, 5, 7, 9, 11, 13]

# Physical error rates (log scale)
# Re=0 threshold ~0.94%, Re=0.98 threshold ~4.15%
p_list_re0 = np.logspace(-3.5, -1, 15).tolist()      # 0.0003 ~ 0.1
p_list_re98 = np.logspace(-2.5, -0.7, 15).tolist()   # 0.003 ~ 0.2

# 시뮬레이션 설정
min_error_cases = 1000   # 논문용이면 10000+
max_repeats = 100000000
time_budget = 600  # 초

# ============== 시뮬레이션 함수 ==============

def simulate_with_Re(p_total, d, Re, noisy_measurements=None):
    """
    Wu et al. 논문 스타일 시뮬레이션
    
    p_total: 총 물리 에러율
    Re: erasure 비율 (0 ~ 1)
    
    p_pauli = p_total * (1 - Re)
    p_erasure = p_total * Re
    """
    if noisy_measurements is None:
        noisy_measurements = d
    
    p_pauli = p_total * (1 - Re)
    p_erasure = p_total * Re
    
    parameters = [
        "--code-type", "rotated-planar-code",
        "--noise-model-builder", "only-gate-error-circuit-level",
        "--noise-model-configuration", '{"use_correlated_pauli":true,"use_correlated_erasure":true}',
        "--decoder", "union-find",
        "--decoder-config", '{"pcmg":true}',
    ]
    
    # erasure가 있으면 pes 추가
    if p_erasure > 0:
        parameters += ["--pes", f"[{p_erasure:.10e}]"]
    
    command = qecp_benchmark_simulate_func_command_vec(
        p_pauli, d, d, noisy_measurements, parameters,
        min_error_cases=min_error_cases,
        max_repeats=max_repeats,
        time_budget=time_budget
    )
    
    stdout, returncode = run_qecp_command_get_stdout(command)
    if returncode != 0:
        print(f"[ERROR] simulation failed for p={p_total}, d={d}, Re={Re}")
        return None, None
    
    # 결과 파싱
    # format: <p> <di> <nm> <shots> <failed> <pL> <dj> <pL_dev> <pe>
    # index:   0    1    2     3       4      5    6      7      8
    full_result = stdout.strip(" \r\n").split("\n")[-1]
    lst = full_result.split(" ")
    pL = float(lst[5])  # error_rate
    pL_dev = float(lst[7])  # confidence_interval
    
    print(f"  d={d:2d}, p={p_total:.4e}, Re={Re}: pL={pL:.4e} ± {pL_dev:.2e}")
    return pL, pL_dev


# ============== 데이터 수집 ==============

def collect_data(Re, p_list):
    """특정 Re 값에 대해 데이터 수집"""
    print("\n" + "="*60)
    print(f"Collecting data for Re = {Re}...")
    print("="*60)
    
    results = {d: {"p": [], "pL": [], "pL_dev": []} for d in code_distances}
    
    for d in code_distances:
        print(f"\nCode distance d={d}:")
        for p in p_list:
            pL, pL_dev = simulate_with_Re(p, d, Re)
            if pL is not None:
                results[d]["p"].append(p)
                results[d]["pL"].append(pL)
                results[d]["pL_dev"].append(pL_dev)
    
    return results


def find_threshold_rough(Re, d_low, d_high, p_start=0.1, p_min=1e-4, decay=0.7):
    """
    Rough estimate로 threshold 위치 찾기
    d_low < d_high인 두 code distance에서 pL이 역전되는 p를 찾음
    
    Returns: estimated threshold
    """
    print(f"\n--- Finding rough threshold for Re={Re} ---")
    print(f"Using d={d_low} and d={d_high}")
    
    p = p_start
    while p > p_min:
        pL_low, _ = simulate_with_Re(p, d_low, Re)
        pL_high, _ = simulate_with_Re(p, d_high, Re)
        
        if pL_low is None or pL_high is None:
            p *= decay
            continue
        
        print(f"  p={p:.4e}: pL(d={d_low})={pL_low:.4e}, pL(d={d_high})={pL_high:.4e}")
        
        # Threshold 아래에서는 d가 클수록 pL이 작아짐
        if pL_low > pL_high:
            print(f"  → Found threshold region around p={p:.4e}")
            return p
        
        p *= decay
    
    print(f"  → Threshold not found, using p_min={p_min}")
    return p_min


def generate_adaptive_p_list(threshold_estimate, n_sparse=5, n_dense=10, 
                              sparse_range=2.0, dense_range=0.5):
    """
    Threshold 근처에서 dense, 멀리서 sparse한 p 리스트 생성
    
    Args:
        threshold_estimate: 추정된 threshold 값
        n_sparse: sparse 영역의 점 개수
        n_dense: dense 영역의 점 개수
        sparse_range: sparse 영역 범위 (log scale에서)
        dense_range: dense 영역 범위 (log scale에서)
    
    Returns:
        정렬된 p 리스트
    """
    log_th = np.log10(threshold_estimate)
    
    # Sparse 영역: threshold 기준 넓은 범위
    p_sparse_low = np.logspace(log_th - sparse_range, log_th - dense_range, n_sparse).tolist()
    p_sparse_high = np.logspace(log_th + dense_range, log_th + sparse_range, n_sparse).tolist()
    
    # Dense 영역: threshold 근처
    p_dense = np.logspace(log_th - dense_range, log_th + dense_range, n_dense).tolist()
    
    # 합치고 정렬, 중복 제거
    all_p = sorted(set(p_sparse_low + p_dense + p_sparse_high))
    
    print(f"\nGenerated {len(all_p)} p values:")
    print(f"  Sparse low:  {p_sparse_low[0]:.2e} ~ {p_sparse_low[-1]:.2e}")
    print(f"  Dense:       {p_dense[0]:.2e} ~ {p_dense[-1]:.2e}")
    print(f"  Sparse high: {p_sparse_high[0]:.2e} ~ {p_sparse_high[-1]:.2e}")
    
    return all_p


def collect_data_adaptive(Re, rough_code_distances=[5, 9], 
                          final_code_distances=None,
                          p_start=0.1, 
                          n_sparse=5, n_dense=12,
                          sparse_range=1.5, dense_range=0.4):
    """
    Adaptive하게 데이터 수집:
    1. 작은 code distance로 threshold 위치 추정
    2. Threshold 근처에서 dense하게 샘플링
    
    Args:
        Re: erasure 비율
        rough_code_distances: threshold 추정에 사용할 [d_low, d_high]
        final_code_distances: 최종 시뮬레이션할 code distances (None이면 global 사용)
        p_start: threshold 탐색 시작점
        n_sparse: sparse 영역 점 개수
        n_dense: dense 영역 점 개수
    """
    global code_distances
    
    if final_code_distances is None:
        final_code_distances = code_distances
    
    d_low, d_high = rough_code_distances
    
    # 1단계: Rough threshold 추정
    print("\n" + "="*60)
    print(f"Phase 1: Rough threshold estimation for Re={Re}")
    print("="*60)
    
    threshold_estimate = find_threshold_rough(Re, d_low, d_high, p_start=p_start)
    print(f"\n>>> Estimated threshold: {threshold_estimate:.4e} ({threshold_estimate*100:.2f}%)")
    
    # 2단계: Adaptive p 리스트 생성
    print("\n" + "="*60)
    print(f"Phase 2: Generate adaptive p list")
    print("="*60)
    
    p_list = generate_adaptive_p_list(
        threshold_estimate, 
        n_sparse=n_sparse, 
        n_dense=n_dense,
        sparse_range=sparse_range,
        dense_range=dense_range
    )
    
    # 3단계: 모든 code distance에 대해 시뮬레이션
    print("\n" + "="*60)
    print(f"Phase 3: Full simulation for Re={Re}")
    print(f"Code distances: {final_code_distances}")
    print(f"p values: {len(p_list)} points")
    print("="*60)
    
    results = {d: {"p": [], "pL": [], "pL_dev": []} for d in final_code_distances}
    
    for d in final_code_distances:
        print(f"\nCode distance d={d}:")
        for p in p_list:
            pL, pL_dev = simulate_with_Re(p, d, Re)
            if pL is not None:
                results[d]["p"].append(p)
                results[d]["pL"].append(pL)
                results[d]["pL_dev"].append(pL_dev)
    
    return results, threshold_estimate


def save_results(results, filename):
    """결과를 JSON으로 저장"""
    # numpy array를 list로 변환
    serializable = {}
    for d, data in results.items():
        serializable[str(d)] = {
            "p": [float(x) for x in data["p"]],
            "pL": [float(x) for x in data["pL"]],
            "pL_dev": [float(x) for x in data["pL_dev"]]
        }
    
    with open(filename, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved results to: {filename}")


def load_results(filename):
    """JSON에서 결과 로드"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    results = {}
    for d_str, values in data.items():
        results[int(d_str)] = values
    return results


# ============== 그래프 그리기 ==============

def plot_wu_fig3a(results_re0, results_re98, save_path="wu_fig3a.pdf"):
    """Wu et al. Figure 3-a 스타일 그래프"""
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # 색상 매핑 (논문과 유사하게)
    colors = {
        3: 'C0',   # 파랑
        5: 'C1',   # 주황
        7: 'C2',   # 초록
        9: 'C3',   # 빨강
        11: 'C4',  # 보라
        13: 'C5',  # 갈색
        15: 'C6',
    }
    
    for d in code_distances:
        color = colors.get(d, 'gray')
        
        # Re = 0 (open circles, dashed lines) - 순수 Pauli
        if d in results_re0 and len(results_re0[d]["p"]) > 0:
            ax.errorbar(
                results_re0[d]["p"],
                results_re0[d]["pL"],
                yerr=results_re0[d]["pL_dev"],
                fmt='o--',
                color=color,
                markerfacecolor='white',  # open circles
                markeredgecolor=color,
                markersize=7,
                capsize=2,
                linewidth=1.5,
                label=f'd = {d}' if d == code_distances[0] else None
            )
        
        # Re = 0.98 (filled circles, solid lines) - 대부분 Erasure
        if d in results_re98 and len(results_re98[d]["p"]) > 0:
            ax.errorbar(
                results_re98[d]["p"],
                results_re98[d]["pL"],
                yerr=results_re98[d]["pL_dev"],
                fmt='o-',
                color=color,
                markerfacecolor=color,  # filled circles
                markersize=7,
                capsize=2,
                linewidth=1.5,
                label=f'd = {d}'
            )
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical error probability, $p$', fontsize=14)
    ax.set_ylabel('Logical error rate, $p_L$', fontsize=14)
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-6, 1)
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # 텍스트 추가
    ax.text(0.02, 0.15, '$R_e = 0$\n(dashed, open)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.text(0.02, 0.05, '$R_e = 0.98$\n(solid, filled)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {save_path}")
    plt.show()


def plot_single(results, Re, save_path="threshold.pdf"):
    """단일 Re 값에 대한 플롯"""
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    colors = {3: 'C0', 5: 'C1', 7: 'C2', 9: 'C3', 11: 'C4', 13: 'C5', 15: 'C6'}
    
    filled = (Re > 0)
    linestyle = '-' if filled else '--'
    
    for d in code_distances:
        if d not in results or len(results[d]["p"]) == 0:
            continue
        
        color = colors.get(d, 'gray')
        
        ax.errorbar(
            results[d]["p"],
            results[d]["pL"],
            yerr=results[d]["pL_dev"],
            fmt=f'o{linestyle}',
            color=color,
            markerfacecolor=color if filled else 'white',
            markeredgecolor=color,
            markersize=7,
            capsize=2,
            linewidth=1.5,
            label=f'd = {d}'
        )
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Physical error probability, $p$', fontsize=14)
    ax.set_ylabel('Logical error rate, $p_L$', fontsize=14)
    ax.set_xlim(1e-4, 1e-1)
    ax.set_ylim(1e-6, 1)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.set_title(f'$R_e = {Re}$', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {save_path}")


# ============== 메인 ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Wu et al. Figure 3-a reproduction')
    parser.add_argument('--mode', choices=['re0', 're98', 'both', 'quick', 'plot', 'adaptive', 'adaptive-quick'], 
                        default='quick', help='Simulation mode')
    parser.add_argument('--output', default='wu_fig3a.pdf', help='Output file path')
    parser.add_argument('--data-dir', default='.', help='Directory for data files')
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    
    compile_code_if_necessary()
    
    if args.mode == 'adaptive-quick':
        # Adaptive 샘플링 (빠른 테스트)
        print("=== Adaptive-Quick mode ===")
        code_distances = [5, 7, 9, 11]
        min_error_cases = 300
        time_budget = 60
        
        # Re=0 adaptive
        print("\n" + "#"*60)
        print("# Re = 0 (Pure Pauli)")
        print("#"*60)
        results_re0, th_re0 = collect_data_adaptive(
            Re=0,
            rough_code_distances=[5, 9],
            final_code_distances=code_distances,
            p_start=0.05,  # Re=0 threshold ~1%
            n_sparse=4, n_dense=8,
            sparse_range=1.2, dense_range=0.35
        )
        save_results(results_re0, os.path.join(args.data_dir, "results_re0_adaptive.json"))
        
        # Re=0.98 adaptive
        print("\n" + "#"*60)
        print("# Re = 0.98 (High Erasure)")
        print("#"*60)
        results_re98, th_re98 = collect_data_adaptive(
            Re=0.98,
            rough_code_distances=[5, 9],
            final_code_distances=code_distances,
            p_start=0.15,  # Re=0.98 threshold ~4%
            n_sparse=4, n_dense=8,
            sparse_range=1.0, dense_range=0.3
        )
        save_results(results_re98, os.path.join(args.data_dir, "results_re98_adaptive.json"))
        
        print(f"\n>>> Threshold estimates: Re=0 → {th_re0*100:.2f}%, Re=0.98 → {th_re98*100:.2f}%")
        
        plot_wu_fig3a(results_re0, results_re98, args.output)
        
    elif args.mode == 'adaptive':
        # Adaptive 샘플링 (논문용, 정밀)
        print("=== Adaptive mode (paper quality) ===")
        code_distances = [5, 7, 9, 11, 13]
        min_error_cases = 1000
        time_budget = 300
        
        # Re=0 adaptive
        print("\n" + "#"*60)
        print("# Re = 0 (Pure Pauli)")
        print("#"*60)
        results_re0, th_re0 = collect_data_adaptive(
            Re=0,
            rough_code_distances=[5, 11],
            final_code_distances=code_distances,
            p_start=0.05,
            n_sparse=6, n_dense=15,
            sparse_range=1.5, dense_range=0.4
        )
        save_results(results_re0, os.path.join(args.data_dir, "results_re0_adaptive.json"))
        
        # Re=0.98 adaptive
        print("\n" + "#"*60)
        print("# Re = 0.98 (High Erasure)")
        print("#"*60)
        results_re98, th_re98 = collect_data_adaptive(
            Re=0.98,
            rough_code_distances=[5, 11],
            final_code_distances=code_distances,
            p_start=0.15,
            n_sparse=6, n_dense=15,
            sparse_range=1.2, dense_range=0.35
        )
        save_results(results_re98, os.path.join(args.data_dir, "results_re98_adaptive.json"))
        
        print(f"\n>>> Threshold estimates: Re=0 → {th_re0*100:.2f}%, Re=0.98 → {th_re98*100:.2f}%")
        
        plot_wu_fig3a(results_re0, results_re98, args.output)
    
    elif args.mode == 'quick':
        # 빠른 테스트용 (개선된 설정)
        print("=== Quick test mode ===")
        code_distances = [5, 7, 9, 11]  # 더 많은 code distance
        
        # Re=0: threshold ~0.9% 근처에 밀집
        p_list_re0 = np.logspace(-2.3, -1.5, 10).tolist()  # 0.5% ~ 3%
        
        # Re=0.98: threshold ~4% 근처에 밀집
        p_list_re98 = np.logspace(-1.7, -1.0, 10).tolist()  # 2% ~ 10%
        
        min_error_cases = 500  # 100 → 500 (분산 줄임)
        time_budget = 120  # 60 → 120초
        
        results_re0 = collect_data(Re=0, p_list=p_list_re0)
        results_re98 = collect_data(Re=0.98, p_list=p_list_re98)
        
        save_results(results_re0, os.path.join(args.data_dir, "results_re0_quick.json"))
        save_results(results_re98, os.path.join(args.data_dir, "results_re98_quick.json"))
        
        plot_wu_fig3a(results_re0, results_re98, args.output)
        
    elif args.mode == 're0':
        # Re = 0 (순수 Pauli) 시뮬레이션
        results_re0 = collect_data(Re=0, p_list=p_list_re0)
        save_results(results_re0, os.path.join(args.data_dir, "results_re0.json"))
        plot_single(results_re0, Re=0, save_path=args.output)
        
    elif args.mode == 're98':
        # Re = 0.98 (대부분 Erasure) 시뮬레이션
        results_re98 = collect_data(Re=0.98, p_list=p_list_re98)
        save_results(results_re98, os.path.join(args.data_dir, "results_re98.json"))
        plot_single(results_re98, Re=0.98, save_path=args.output)
        
    elif args.mode == 'both':
        # 둘 다 시뮬레이션 (시간 오래 걸림)
        results_re0 = collect_data(Re=0, p_list=p_list_re0)
        save_results(results_re0, os.path.join(args.data_dir, "results_re0.json"))
        
        results_re98 = collect_data(Re=0.98, p_list=p_list_re98)
        save_results(results_re98, os.path.join(args.data_dir, "results_re98.json"))
        
        plot_wu_fig3a(results_re0, results_re98, args.output)
        
    elif args.mode == 'plot':
        # 기존 데이터로 플롯만
        try:
            results_re0 = load_results(os.path.join(args.data_dir, "results_re0_adaptive.json"))
            results_re98 = load_results(os.path.join(args.data_dir, "results_re98_adaptive.json"))
        except FileNotFoundError:
            results_re0 = load_results(os.path.join(args.data_dir, "results_re0_quick.json"))
            results_re98 = load_results(os.path.join(args.data_dir, "results_re98_quick.json"))
        plot_wu_fig3a(results_re0, results_re98, args.output)