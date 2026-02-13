"""
Utility functions for threshold analysis

이 모듈은 여러 threshold 분석 스크립트에서 공유하는 유틸리티 함수들을 포함합니다.
"""

import os
import time
import threading
import numpy as np
import matplotlib.pyplot as plt


# ============== Progress Tracking ==============

def format_duration(seconds):
    """Format seconds into human-readable duration string."""
    if seconds < 0:
        return "0s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


class ProgressTracker:
    """Track simulation progress and estimate remaining time.

    Thread-safe: works in both sequential and parallel modes.

    Usage (sequential):
        tracker = ProgressTracker(total_tasks, "simulations", print_every=4)
        for ...:
            tracker.begin_task()
            # ... do work ...
            tracker.end_task()
        tracker.summary()

    Usage (parallel):
        tracker = ProgressTracker(total_tasks, "simulations")
        # in each worker thread, after task completes:
        tracker.task_done()
        tracker.summary()
    """

    def __init__(self, total_tasks, description="simulations", print_every=None):
        self.total_tasks = total_tasks
        self.completed = 0
        self.start_time = time.time()
        self.task_times = []
        self.description = description
        self._task_start = None
        self._lock = threading.Lock()
        if print_every is None:
            self.print_every = max(1, total_tasks // 20)
        else:
            self.print_every = max(1, print_every)

    def begin_task(self):
        """Call before starting a task (sequential mode only)."""
        self._task_start = time.time()

    def end_task(self):
        """Call after completing a task (sequential mode). Prints progress periodically."""
        with self._lock:
            if self._task_start is not None:
                self.task_times.append(time.time() - self._task_start)
                self._task_start = None
            self.completed += 1
            should_print = (self.completed % self.print_every == 0 or
                    self.completed == self.total_tasks or
                    self.completed == 1)
        if should_print:
            self._print_progress()

    def task_done(self):
        """Thread-safe: increment completion counter (parallel mode)."""
        with self._lock:
            self.completed += 1
            should_print = (self.completed % self.print_every == 0 or
                    self.completed == self.total_tasks or
                    self.completed == 1)
        if should_print:
            self._print_progress()

    def _print_progress(self):
        with self._lock:
            elapsed = time.time() - self.start_time
            completed = self.completed
            remaining_tasks = self.total_tasks - completed
            pct = completed / self.total_tasks * 100
            has_task_times = len(self.task_times) > 0
            avg_task = sum(self.task_times) / len(self.task_times) if has_task_times else 0

        if has_task_times:
            eta = avg_task * remaining_tasks
            print(f"  \u23f1  [{completed}/{self.total_tasks}] ({pct:.0f}%) "
                  f"elapsed: {format_duration(elapsed)}, "
                  f"ETA: ~{format_duration(eta)}, "
                  f"avg: {avg_task:.1f}s/sim")
        elif completed > 0 and elapsed > 0:
            # parallel mode: estimate from throughput
            throughput = completed / elapsed
            eta = remaining_tasks / throughput if throughput > 0 else 0
            print(f"  \u23f1  [{completed}/{self.total_tasks}] ({pct:.0f}%) "
                  f"elapsed: {format_duration(elapsed)}, "
                  f"ETA: ~{format_duration(eta)}, "
                  f"throughput: {throughput:.1f} sims/s")
        else:
            print(f"  \u23f1  [{completed}/{self.total_tasks}] ({pct:.0f}%) "
                  f"elapsed: {format_duration(elapsed)}")

    def summary(self):
        """Print final summary when all tasks are done."""
        total_time = time.time() - self.start_time
        if self.task_times:
            avg = sum(self.task_times) / len(self.task_times)
            print(f"  \u23f1  Done: {self.completed} {self.description} "
                  f"in {format_duration(total_time)} (avg {avg:.1f}s/sim)")
        else:
            throughput = self.completed / total_time if total_time > 0 else 0
            print(f"  \u23f1  Done: {self.completed} {self.description} "
                  f"in {format_duration(total_time)} ({throughput:.1f} sims/s)")


def scaled_runtime_budget(runtime_budget, d, d_base):
    """Scale runtime_budget according to code distance.

    Larger code distances need proportionally more time because:
    - Each simulation trial is slower (more qubits & rounds)
    - Logical error rates are lower → need more trials to collect errors

    Scaling: time_budget *= (d / d_base)^2
             min_error_cases stays the same (we want the same statistical quality)

    Args:
        runtime_budget: (min_error_cases, time_budget)
        d: current code distance
        d_base: smallest code distance (reference)

    Returns:
        (min_error_cases, scaled_time_budget)
    """
    min_error_cases, time_budget = runtime_budget
    scale = (d / d_base) ** 2
    return (min_error_cases, int(time_budget * scale))


def run_parallel_simulations(simulate_func, code_distances, p_list, runtime_budget, n_workers):
    """Run p-sweep simulations in parallel using ThreadPoolExecutor.

    Each (p, d) pair is submitted as an independent task.
    Results are collected and organized in the same order as sequential execution.

    Args:
        simulate_func: function(p, d, runtime_budget, p_graph) -> (pL, pL_dev)
        code_distances: list of code distances
        p_list: list of physical error rates
        runtime_budget: (min_error_cases, time_budget)
        n_workers: number of parallel threads

    Returns:
        {d: {"p": [...], "pL": [...], "pL_dev": [...]}}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {d: {"p": [], "pL": [], "pL_dev": []} for d in code_distances}
    task_list = [(p, d) for p in p_list for d in code_distances]
    total = len(task_list)

    print(f"  \u26a1 Parallel mode: {n_workers} workers, {total} tasks")
    d_base = min(code_distances)
    tracker = ProgressTracker(total, "simulations", print_every=max(1, n_workers))
    result_map = {}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_key = {}
        for p, d in task_list:
            budget = scaled_runtime_budget(runtime_budget, d, d_base)
            future = executor.submit(simulate_func, p, d, budget, p)
            future_to_key[future] = (p, d)

        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                pL, pL_dev = future.result()
            except Exception as e:
                print(f"  [ERROR] p={key[0]:.4e}, d={key[1]}: {e}")
                pL, pL_dev = 0.5, 1.0
            result_map[key] = (pL, pL_dev)
            tracker.task_done()

    # Organize results in original p-order
    for p in p_list:
        for d in code_distances:
            pL, pL_dev = result_map[(p, d)]
            results[d]["p"].append(p)
            results[d]["pL"].append(pL)
            results[d]["pL_dev"].append(pL_dev)

    tracker.summary()
    return results


def save_checkpoint(results, checkpoint_path):
    """중간 결과를 checkpoint 파일로 저장.

    각 p값 시뮬레이션 완료 후 호출하여 진행 상황을 보존합니다.

    Args:
        results: {d: {"p": [...], "pL": [...], "pL_dev": [...]}}
        checkpoint_path: checkpoint JSON 파일 경로
    """
    import json as _json
    data = {}
    for d, vals in results.items():
        data[str(d)] = {
            "p": [float(x) for x in vals["p"]],
            "pL": [float(x) for x in vals["pL"]],
            "pL_dev": [float(x) for x in vals["pL_dev"]]
        }
    tmp_path = checkpoint_path + ".tmp"
    with open(tmp_path, 'w') as f:
        _json.dump(data, f, indent=2)
    os.replace(tmp_path, checkpoint_path)


def clean_checkpoints(data_dir):
    """data_dir 내의 모든 checkpoint_*.json 파일을 삭제합니다.

    --fresh 옵션과 함께 사용하여 처음부터 다시 시뮬레이션을 시작할 때 사용합니다.
    """
    import glob
    pattern = os.path.join(data_dir, "checkpoint_*.json")
    files = glob.glob(pattern)
    if files:
        for f in files:
            os.remove(f)
            print(f"  🗑  Removed checkpoint: {os.path.basename(f)}")
        print(f"  Cleaned {len(files)} checkpoint file(s) from {data_dir}")


def load_checkpoint(checkpoint_path, code_distances):
    """Checkpoint 파일에서 이전 결과 로드.

    Args:
        checkpoint_path: checkpoint JSON 파일 경로
        code_distances: 예상되는 code distance 목록

    Returns:
        results: {d: {"p": [...], "pL": [...], "pL_dev": [...]}} 또는 빈 dict
        completed_p: set of p values that have ALL code distances completed
    """
    import json as _json
    results = {d: {"p": [], "pL": [], "pL_dev": []} for d in code_distances}
    completed_p = set()

    if not os.path.exists(checkpoint_path):
        return results, completed_p

    try:
        with open(checkpoint_path, 'r') as f:
            data = _json.load(f)

        for d_str, vals in data.items():
            d = int(d_str)
            if d in results:
                results[d] = {
                    "p": list(vals["p"]),
                    "pL": list(vals["pL"]),
                    "pL_dev": list(vals["pL_dev"]),
                }

        # 모든 code distance에서 완료된 p값 찾기
        p_sets = []
        for d in code_distances:
            p_sets.append(set(round(p, 10) for p in results[d]["p"]))
        if p_sets:
            completed_p = p_sets[0]
            for ps in p_sets[1:]:
                completed_p = completed_p & ps

        n_total = sum(len(results[d]["p"]) for d in code_distances)
        print(f"  📂 Loaded checkpoint: {checkpoint_path} ({n_total} data points, {len(completed_p)} p-values complete)")
    except Exception as e:
        print(f"  ⚠️  Failed to load checkpoint {checkpoint_path}: {e}")
        results = {d: {"p": [], "pL": [], "pL_dev": []} for d in code_distances}
        completed_p = set()

    return results, completed_p


def run_p_sweep_with_checkpoint(simulate_func, code_distances, p_list, runtime_budget,
                                checkpoint_path=None, n_workers=1):
    """run_p_sweep with checkpoint/resume support.

    각 p값에 대한 모든 code distance 시뮬레이션 완료 후 자동 저장합니다.
    이전에 저장된 checkpoint가 있으면 완료된 p값을 건너뜁니다.

    Args:
        simulate_func: function(p, d, runtime_budget, p_graph) -> (pL, pL_dev)
        code_distances: list of code distances
        p_list: list of physical error rates
        runtime_budget: (min_error_cases, time_budget)
        checkpoint_path: checkpoint 파일 경로 (None이면 checkpoint 비활성화)
        n_workers: 병렬 워커 수 (1 = 순차)

    Returns:
        {d: {"p": [...], "pL": [...], "pL_dev": [...]}}
    """
    # Load existing checkpoint
    if checkpoint_path:
        results, completed_p = load_checkpoint(checkpoint_path, code_distances)
        remaining_p = [p for p in p_list if round(p, 10) not in completed_p]
        if len(remaining_p) < len(p_list):
            skipped = len(p_list) - len(remaining_p)
            print(f"  ⏩ Resuming: skipping {skipped}/{len(p_list)} already-completed p values")
        if not remaining_p:
            print(f"  ✅ All p values already completed in checkpoint")
            return results
    else:
        results = {d: {"p": [], "pL": [], "pL_dev": []} for d in code_distances}
        remaining_p = list(p_list)

    if n_workers > 1 and len(remaining_p) > 0:
        # Parallel mode with periodic checkpoint
        new_results = _run_parallel_with_checkpoint(
            simulate_func, code_distances, remaining_p, runtime_budget,
            n_workers, checkpoint_path, results)
        return new_results

    # Sequential mode
    total_sims = len(remaining_p) * len(code_distances)
    if total_sims == 0:
        return results

    tracker = ProgressTracker(total_sims, "simulations", print_every=len(code_distances))
    d_base = min(code_distances)

    for p in remaining_p:
        print(f"\n--- p = {p:.4e} ---")
        for d in code_distances:
            tracker.begin_task()
            pL, pL_dev = simulate_func(p, d, scaled_runtime_budget(runtime_budget, d, d_base), p_graph=p)
            results[d]["p"].append(p)
            results[d]["pL"].append(pL)
            results[d]["pL_dev"].append(pL_dev)
            tracker.end_task()
        # Checkpoint after each p value completes all code distances
        if checkpoint_path:
            save_checkpoint(results, checkpoint_path)

    tracker.summary()
    return results


def _run_parallel_with_checkpoint(simulate_func, code_distances, p_list, runtime_budget,
                                  n_workers, checkpoint_path, results):
    """Parallel p-sweep with checkpoint support.

    p값 단위로 그룹화하여, 한 p의 모든 d가 끝나면 checkpoint를 저장합니다.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    d_base = min(code_distances)
    total = len(p_list) * len(code_distances)
    print(f"  ⚡ Parallel mode: {n_workers} workers, {total} tasks")
    tracker = ProgressTracker(total, "simulations", print_every=max(1, n_workers))

    lock = threading.Lock()
    # Track completed (p, d) pairs for this batch
    p_results = {}  # {round(p, 10): {d: (pL, pL_dev)}}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_key = {}
        for p in p_list:
            for d in code_distances:
                budget = scaled_runtime_budget(runtime_budget, d, d_base)
                future = executor.submit(simulate_func, p, d, budget, p)
                future_to_key[future] = (p, d)

        for future in as_completed(future_to_key):
            p, d = future_to_key[future]
            try:
                pL, pL_dev = future.result()
            except Exception as e:
                print(f"  [ERROR] p={p:.4e}, d={d}: {e}")
                pL, pL_dev = 0.5, 1.0

            p_key = round(p, 10)
            with lock:
                if p_key not in p_results:
                    p_results[p_key] = {}
                p_results[p_key][d] = (pL, pL_dev)

                # Check if all code distances for this p are done
                if len(p_results[p_key]) == len(code_distances):
                    for dd in code_distances:
                        pL_dd, pL_dev_dd = p_results[p_key][dd]
                        results[dd]["p"].append(p)
                        results[dd]["pL"].append(pL_dd)
                        results[dd]["pL_dev"].append(pL_dev_dd)
                    if checkpoint_path:
                        save_checkpoint(results, checkpoint_path)

            tracker.task_done()

    tracker.summary()
    return results


def find_crossing_point(results, d1, d2, verbose=False):
    """
    두 code distance의 데이터로부터 교차점 찾기 (실제 데이터 포인트 사용)
    
    각 code distance의 pL(p) 곡선에서 공통 p 값을 찾고,
    diff = log(pL(d_small)) - log(pL(d_large))가 양수에서 음수로 바뀌는 지점을 찾음.
    
    Args:
        results: {d: {"p": [...], "pL": [...], "pL_dev": [...]}}
        d1: 첫 번째 code distance (작은 값)
        d2: 두 번째 code distance (큰 값)
        verbose: 디버깅 출력 여부
    
    Returns:
        crossing_p: 교차점의 p 값 (없으면 None)
    """
    if d1 not in results or d2 not in results:
        if verbose:
            print(f"    [find_crossing] d1={d1} or d2={d2} not in results")
        return None
    
    # d1 < d2 보장
    if d1 > d2:
        d1, d2 = d2, d1
    
    p1 = np.array(results[d1]["p"])
    pL1 = np.array(results[d1]["pL"])
    p2 = np.array(results[d2]["p"])
    pL2 = np.array(results[d2]["pL"])
    
    # 공통 p 값 찾기 (float 비교를 위해 tolerance 사용)
    common_p = []
    for p in p1:
        matches = np.where(np.isclose(p2, p, rtol=1e-6))[0]
        if len(matches) > 0:
            common_p.append(p)
    common_p = sorted(common_p)
    
    if len(common_p) < 2:
        if verbose:
            print(f"    [find_crossing] Not enough common p values: {len(common_p)}")
        return None
    
    if verbose:
        print(f"    [find_crossing] d1={d1}, d2={d2}: {len(common_p)} common p values")
    
    # 각 공통 p에서 diff 계산
    prev_diff = None
    prev_p = None
    
    for p in common_p:
        idx1 = np.where(np.isclose(p1, p, rtol=1e-6))[0][0]
        idx2 = np.where(np.isclose(p2, p, rtol=1e-6))[0][0]
        
        # pL > 0인 경우만 사용
        if pL1[idx1] <= 0 or pL2[idx2] <= 0:
            continue
        
        diff = np.log(pL1[idx1]) - np.log(pL2[idx2])
        
        if verbose:
            print(f"    [find_crossing] p={p:.6f}: pL({d1})={pL1[idx1]:.6f}, pL({d2})={pL2[idx2]:.6f}, diff={diff:.4f}")
        
        # 양수 → 음수 전환 확인 (threshold 통과)
        if prev_diff is not None and prev_diff > 0 and diff < 0:
            # 두 점 사이에서 선형 보간
            p_cross = prev_p + (p - prev_p) * prev_diff / (prev_diff - diff)
            if verbose:
                print(f"    [find_crossing] CROSSING found between p={prev_p:.6f} and p={p:.6f}")
                print(f"    [find_crossing] p_cross = {p_cross:.6f} ({p_cross*100:.3f}%)")
            return p_cross
        
        prev_diff = diff
        prev_p = p
    
    if verbose:
        print(f"    [find_crossing] No crossing found (diff never went from + to -)")
    return None


def estimate_threshold_from_data(results, code_distances, verbose=True, method="adjacent"):
    """
    데이터로부터 threshold 추정
    
    Args:
        results: {d: {"p": [...], "pL": [...], "pL_dev": [...]}}
        code_distances: 사용할 code distance 목록
        verbose: 상세 출력 여부
        method: 추정 방법
            - "adjacent": 인접한 쌍들의 평균 (기본, 가장 안정적)
            - "largest_pair": 가장 큰 두 code distance의 교차점 (유한 크기 효과 최소)
            - "smallest_pair": 가장 작은 두 code distance의 교차점 (데이터 범위가 좁을 때)
            - "all_pairs": 모든 쌍의 교차점 중 median
    
    Returns:
        threshold: 추정된 threshold 값
        threshold_err: 표준편차 또는 추정 오차
    """
    sorted_d = sorted(code_distances)
    
    if method == "largest_pair":
        # 가장 큰 두 code distance 사용 (유한 크기 효과 최소화)
        if len(sorted_d) < 2:
            return None, None
        d1, d2 = sorted_d[-2], sorted_d[-1]
        crossing = find_crossing_point(results, d1, d2, verbose=verbose)
        if crossing is not None and crossing > 0:
            if verbose:
                print(f"  Crossing d={d1} & d={d2}: p = {crossing:.6f}")
            # 오차 추정: 그 다음 큰 쌍과의 차이
            if len(sorted_d) >= 3:
                d0 = sorted_d[-3]
                crossing2 = find_crossing_point(results, d0, d2, verbose=False)
                if crossing2 is not None:
                    threshold_err = abs(crossing - crossing2)
                else:
                    threshold_err = crossing * 0.1
            else:
                threshold_err = crossing * 0.1
            if verbose:
                print(f"  => Estimated threshold: {crossing:.6f} ± {threshold_err:.6f}")
            return crossing, threshold_err
        
        # largest_pair에서 못 찾으면 smallest_pair 시도
        if verbose:
            print(f"  [No crossing found with largest pair, trying smallest pair]")
        d1, d2 = sorted_d[0], sorted_d[1]
        crossing = find_crossing_point(results, d1, d2, verbose=verbose)
        if crossing is not None and crossing > 0:
            if verbose:
                print(f"  Crossing d={d1} & d={d2}: p = {crossing:.6f}")
            threshold_err = crossing * 0.15  # 작은 d는 유한 크기 효과가 커서 오차 더 큼
            if verbose:
                print(f"  => Estimated threshold: {crossing:.6f} ± {threshold_err:.6f}")
            return crossing, threshold_err
        
        return None, None
    
    elif method == "smallest_pair":
        # 가장 작은 두 code distance 사용 (데이터 범위가 좁을 때 유용)
        if len(sorted_d) < 2:
            return None, None
        d1, d2 = sorted_d[0], sorted_d[1]
        crossing = find_crossing_point(results, d1, d2, verbose=verbose)
        if crossing is not None and crossing > 0:
            if verbose:
                print(f"  Crossing d={d1} & d={d2}: p = {crossing:.6f}")
            threshold_err = crossing * 0.15
            if verbose:
                print(f"  => Estimated threshold: {crossing:.6f} ± {threshold_err:.6f}")
            return crossing, threshold_err
        return None, None
    
    elif method == "all_pairs":
        # 모든 쌍의 교차점 계산 후 median
        crossings = []
        for i in range(len(sorted_d)):
            for j in range(i + 1, len(sorted_d)):
                d1, d2 = sorted_d[i], sorted_d[j]
                crossing = find_crossing_point(results, d1, d2, verbose=verbose)
                if crossing is not None and crossing > 0:
                    if verbose:
                        print(f"  Crossing d={d1} & d={d2}: p = {crossing:.6f}")
                    crossings.append(crossing)
        
        if not crossings:
            return None, None
        
        threshold = np.median(crossings)
        threshold_err = np.std(crossings) if len(crossings) > 1 else threshold * 0.1
        if verbose:
            print(f"  => Estimated threshold (median of {len(crossings)} pairs): {threshold:.6f} ± {threshold_err:.6f}")
        return threshold, threshold_err
    
    else:  # "adjacent" - 기존 방식
        crossings = []
        for i in range(len(sorted_d) - 1):
            d1, d2 = sorted_d[i], sorted_d[i + 1]
            crossing = find_crossing_point(results, d1, d2, verbose=verbose)
            if crossing is not None and crossing > 0:
                if verbose:
                    print(f"  Crossing d={d1} & d={d2}: p = {crossing:.6f}")
                crossings.append(crossing)
        
        if not crossings:
            return None, None
        
        threshold = np.mean(crossings)
        threshold_err = np.std(crossings) if len(crossings) > 1 else threshold * 0.1
        
        if verbose:
            print(f"  => Estimated threshold: {threshold:.6f} ± {threshold_err:.6f}")
        return threshold, threshold_err


def merge_results(results1, results2):
    """
    두 결과 dict를 병합 (중복 p 제거)
    
    Args:
        results1: 첫 번째 결과 dict
        results2: 두 번째 결과 dict
    
    Returns:
        merged: 병합된 결과 dict
    """
    merged = {}
    all_d = set(results1.keys()) | set(results2.keys())
    
    for d in all_d:
        merged[d] = {"p": [], "pL": [], "pL_dev": []}
        
        # 두 결과를 합침
        seen_p = set()
        for res in [results1, results2]:
            if d in res:
                for i, p in enumerate(res[d]["p"]):
                    p_key = round(p, 8)  # 부동소수점 비교용
                    if p_key not in seen_p:
                        seen_p.add(p_key)
                        merged[d]["p"].append(p)
                        merged[d]["pL"].append(res[d]["pL"][i])
                        merged[d]["pL_dev"].append(res[d]["pL_dev"][i])
        
        # p 기준 정렬
        if merged[d]["p"]:
            sorted_idx = np.argsort(merged[d]["p"])
            merged[d]["p"] = [merged[d]["p"][i] for i in sorted_idx]
            merged[d]["pL"] = [merged[d]["pL"][i] for i in sorted_idx]
            merged[d]["pL_dev"] = [merged[d]["pL_dev"][i] for i in sorted_idx]
    
    return merged


# ============== Lambda Factor ==============

def compute_lambda_factor(results, code_distances, p_fixed=None):
    """
    Compute Lambda factor: Λ(p) = pL(d_small) / pL(d_large).

    At a fixed physical error rate p, Λ > 1 means error suppression
    is working (larger code → lower logical error rate).

    Args:
        results: {d: {"p": [...], "pL": [...], "pL_dev": [...]}}
        code_distances: list of code distances
        p_fixed: if given, only compute Λ at this p; otherwise compute for all p

    Returns:
        lambda_data: {(d_small, d_large): {"p": [...], "lambda": [...], "lambda_err": [...]}}
    """
    sorted_d = sorted(code_distances)
    lambda_data = {}

    for i in range(len(sorted_d) - 1):
        d_small = sorted_d[i]
        d_large = sorted_d[i + 1]
        if d_small not in results or d_large not in results:
            continue

        p_small = np.array(results[d_small]["p"])
        pL_small = np.array(results[d_small]["pL"])
        pL_dev_small = np.array(results[d_small]["pL_dev"])

        p_large = np.array(results[d_large]["p"])
        pL_large = np.array(results[d_large]["pL"])
        pL_dev_large = np.array(results[d_large]["pL_dev"])

        pair_key = (d_small, d_large)
        lambda_data[pair_key] = {"p": [], "lambda": [], "lambda_err": []}

        for p_val in p_small:
            if p_fixed is not None and not np.isclose(p_val, p_fixed, rtol=1e-4):
                continue
            idx_l = np.where(np.isclose(p_large, p_val, rtol=1e-6))[0]
            if len(idx_l) == 0:
                continue
            idx_s = np.where(np.isclose(p_small, p_val, rtol=1e-6))[0][0]
            idx_l = idx_l[0]

            pL_s = pL_small[idx_s]
            pL_l = pL_large[idx_l]

            if pL_s <= 0 or pL_l <= 0:
                continue

            lam = pL_s / pL_l
            # Error propagation: Λ = a/b → δΛ/Λ = sqrt((δa/a)² + (δb/b)²)
            rel_err_s = pL_dev_small[idx_s] / pL_s if pL_s > 0 else 0
            rel_err_l = pL_dev_large[idx_l] / pL_l if pL_l > 0 else 0
            lam_err = lam * np.sqrt(rel_err_s**2 + rel_err_l**2)

            lambda_data[pair_key]["p"].append(p_val)
            lambda_data[pair_key]["lambda"].append(lam)
            lambda_data[pair_key]["lambda_err"].append(lam_err)

    return lambda_data


def print_lambda_summary(lambda_data, label=""):
    """Print Λ-factor table."""
    if label:
        print(f"\n{'='*70}")
        print(f"  Λ-factor: {label}")
        print(f"{'='*70}")

    for pair_key in sorted(lambda_data.keys()):
        d_s, d_l = pair_key
        ld = lambda_data[pair_key]
        if not ld["p"]:
            continue
        print(f"\n  d={d_s} → d={d_l}:")
        print(f"  {'p':>12s}  {'Λ':>10s}  {'± err':>10s}  {'status':>10s}")
        print(f"  {'-'*48}")
        for j in range(len(ld["p"])):
            p_val = ld["p"][j]
            lam = ld["lambda"][j]
            lam_err = ld["lambda_err"][j]
            status = "✅ Λ>1" if lam > 1 else "❌ Λ≤1"
            print(f"  {p_val:>12.4e}  {lam:>10.3f}  {lam_err:>10.3f}  {status:>10s}")


def plot_lambda_comparison(lambda_datasets, code_distances,
                           title="", save_path="lambda_comparison.pdf"):
    """
    Plot Λ(p) for multiple scenarios on separate subplots for comparison.

    Each distance pair gets ONE subplot; all scenarios (e.g. soft vs no-erasure)
    are overlaid as separate lines on that subplot.

    Args:
        lambda_datasets: list of (label, color, marker, linestyle, lambda_data) tuples
            lambda_data = {(d_small, d_large): {"p": [...], "lambda": [...], "lambda_err": [...]}}
        code_distances: list of code distances
        title: overall figure title
        save_path: output file path
    """
    sorted_d = sorted(code_distances)
    all_pairs = [(sorted_d[i], sorted_d[i + 1]) for i in range(len(sorted_d) - 1)]
    n_pairs = len(all_pairs)
    if n_pairs == 0:
        print("Not enough code distances to plot Λ")
        return

    fig, axes = plt.subplots(1, n_pairs, figsize=(5.5 * n_pairs, 5), squeeze=False)

    for i, pair_key in enumerate(all_pairs):
        d_s, d_l = pair_key
        ax = axes[0][i]

        for label, color, marker, ls, lam_data in lambda_datasets:
            if pair_key not in lam_data:
                continue
            ld = lam_data[pair_key]
            if not ld["p"]:
                continue
            ax.errorbar(ld["p"], ld["lambda"], yerr=ld["lambda_err"],
                        fmt=marker, linestyle=ls, color=color,
                        markersize=5, linewidth=1.5,
                        label=label, capsize=3,
                        markerfacecolor=color if '-' in ls else 'white',
                        markeredgecolor=color)

        ax.axhline(y=1, color='red', linestyle=':', alpha=0.5, linewidth=1)
        ax.set_xscale('log')
        ax.set_xlabel('Physical error rate $p$', fontsize=12)
        ax.set_ylabel(f'$\\Lambda_{{d={d_s}\\to{d_l}}}$', fontsize=14)
        ax.set_title(f'd={d_s} → d={d_l}', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved Λ-factor plot: {save_path}")
    plt.show()


def resolve_parallel_workers(n):
    """Resolve --parallel argument: 0 means use all CPU cores.

    Args:
        n: user-specified worker count (0 = all cores, 1 = sequential)

    Returns:
        int: actual number of workers
    """
    if n <= 0:
        import os as _os
        n = _os.cpu_count() or 1  # None on rare platforms → fallback to 1
        print(f"  ⚡ --parallel 0 → using all {n} CPU cores")
    return n


def save_lambda_results(lambda_data, filename):
    """Save Λ-factor data to JSON.

    Args:
        lambda_data: {(d_small, d_large): {"p": [...], "lambda": [...], "lambda_err": [...]}}
        filename: output JSON path
    """
    import json as _json
    serializable = {}
    for (d_s, d_l), vals in lambda_data.items():
        key = f"{d_s}_{d_l}"
        serializable[key] = {
            "d_small": d_s,
            "d_large": d_l,
            "p": [float(x) for x in vals["p"]],
            "lambda": [float(x) for x in vals["lambda"]],
            "lambda_err": [float(x) for x in vals["lambda_err"]],
        }
    with open(filename, 'w') as f:
        _json.dump(serializable, f, indent=2)
    print(f"Saved Λ-factor data: {filename}")


def load_lambda_results(filename):
    """Load Λ-factor data from JSON.

    Returns:
        {(d_small, d_large): {"p": [...], "lambda": [...], "lambda_err": [...]}}
    """
    import json as _json
    with open(filename, 'r') as f:
        data = _json.load(f)
    lambda_data = {}
    for key, vals in data.items():
        pair = (vals["d_small"], vals["d_large"])
        lambda_data[pair] = {
            "p": vals["p"],
            "lambda": vals["lambda"],
            "lambda_err": vals["lambda_err"],
        }
    return lambda_data
