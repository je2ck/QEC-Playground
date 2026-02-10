#!/usr/bin/env python3
"""
Calculate multi-class erasure parameters (Rm_k, Rc_k, Bayes weight) from:
  1. Overall confusion matrix CSV (Den row) - for total error/correct counts
  2. Ambiguous zone CSV (per-sample with WeightClass) - for per-class counts

Outputs simulator config JSON for each exposure time.
"""

import csv
import json
import os
import sys
from collections import defaultdict

# Paths
OVERALL_CSV = "/Users/jaeickbae/Desktop/My paper/erasure-conversion-readout/erasure-stats/시트 1-표 1.csv"
AMB_DIR = "/Users/jaeickbae/Documents/research/denoise/Noise2NoiseFlow/noise2noiseflow/uncertainty_weighted_outputs"

def parse_overall_csv(path):
    """Parse the overall confusion matrix CSV. Returns dict: exposure -> {method -> {TP, TN, FP, FN}}"""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp = int(row['exposure'])
            method = row['Method']
            if exp not in data:
                data[exp] = {}
            data[exp][method] = {
                'TP': int(row['TP']),
                'TN': int(row['TN']),
                'FP': int(row['FP']),
                'FN': int(row['FN']),
            }
    return data

def parse_amb_csv(path):
    """Parse ambiguous zone CSV. Returns per-class (L/R merged) counts: {class -> {TP, TN, FP, FN}}"""
    counts = defaultdict(lambda: defaultdict(int))
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cls = row['WeightClass'][:2]  # C1, C2, C3, C4 (merge L/R)
            cat = row['Category']
            counts[cls][cat] += 1
    return counts

def compute_erasure_classes(den_stats, amb_counts):
    """
    Compute Rm_k, Rc_k, and Bayes weight for each class.
    
    den_stats: {TP, TN, FP, FN} from Den row (total)
    amb_counts: {class -> {TP, TN, FP, FN}} from ambiguous zone CSV
    
    Returns: (Pm, classes_list)
    """
    total_wrong = den_stats['FP'] + den_stats['FN']      # total measurement errors
    total_correct = den_stats['TP'] + den_stats['TN']    # total correct measurements
    total_samples = total_wrong + total_correct
    Pm = total_wrong / total_samples if total_samples > 0 else 0
    
    classes = []
    for cls in sorted(amb_counts):
        tp = amb_counts[cls]['TP']
        tn = amb_counts[cls]['TN']
        fp = amb_counts[cls]['FP']
        fn = amb_counts[cls]['FN']
        
        wrong_k = fp + fn       # measurement errors in this class
        correct_k = tp + tn     # correct measurements in this class
        
        Rm_k = wrong_k / total_wrong if total_wrong > 0 else 0
        Rc_k = correct_k / total_correct if total_correct > 0 else 0
        
        # Bayes weight: P(error | class k erasure) = Pm * Rm_k / (Pm * Rm_k + (1-Pm) * Rc_k)
        numerator = Pm * Rm_k
        denominator = Pm * Rm_k + (1 - Pm) * Rc_k
        weight_k = numerator / denominator if denominator > 0 else 0
        
        classes.append({
            'name': cls,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'wrong': wrong_k, 'correct': correct_k,
            'Rm': Rm_k,
            'Rc': Rc_k,
            'weight': weight_k,
        })
    
    return Pm, classes

def main():
    overall_data = parse_overall_csv(OVERALL_CSV)
    
    for exposure in sorted(overall_data.keys(), reverse=True):
        if 'Den' not in overall_data[exposure]:
            continue
        
        amb_csv = os.path.join(AMB_DIR, f"ambiguous_zone_data_{exposure}ms.csv")
        if not os.path.exists(amb_csv):
            print(f"\n=== {exposure}ms: No ambiguous zone CSV found, skipping ===")
            continue
        
        den_stats = overall_data[exposure]['Den']
        amb_counts = parse_amb_csv(amb_csv)
        
        if not amb_counts:
            print(f"\n=== {exposure}ms: Empty ambiguous zone data, skipping ===")
            continue
        
        Pm, classes = compute_erasure_classes(den_stats, amb_counts)
        
        total_wrong = den_stats['FP'] + den_stats['FN']
        total_correct = den_stats['TP'] + den_stats['TN']
        total_samples = total_wrong + total_correct
        
        sum_Rm = sum(c['Rm'] for c in classes)
        sum_Rc = sum(c['Rc'] for c in classes)
        
        print(f"\n{'='*80}")
        print(f"  {exposure}ms exposure")
        print(f"{'='*80}")
        print(f"  Den: TP={den_stats['TP']}, TN={den_stats['TN']}, FP={den_stats['FP']}, FN={den_stats['FN']}")
        print(f"  Total samples: {total_samples}")
        print(f"  Pm = {Pm:.6f} ({total_wrong} errors / {total_samples} total)")
        print(f"  Sum(Rm_k) = {sum_Rm:.4f}  (1 - Sum = {1-sum_Rm:.4f} -> hidden errors)")
        print(f"  Sum(Rc_k) = {sum_Rc:.4f}  (1 - Sum = {1-sum_Rc:.4f} -> no erasure)")
        print()
        
        header = f"  {'Class':<6} {'Wrong':>6} {'Correct':>8} {'Rm_k':>8} {'Rc_k':>8} {'Weight':>8}"
        print(header)
        print(f"  {'-'*len(header)}")
        for c in classes:
            print(f"  {c['name']:<6} {c['wrong']:>6} {c['correct']:>8} {c['Rm']:>8.4f} {c['Rc']:>8.4f} {c['weight']:>8.4f}")
        
        # Generate simulator config JSON (multi-class)
        config = {
            "measurement_error_rate_total": Pm,
            "erasure_classes": [
                {"Rm": round(c['Rm'], 6), "Rc": round(c['Rc'], 6)}
                for c in classes
            ]
        }
        
        print(f"\n  Multi-class config:")
        print(f"  {json.dumps(config, indent=4)}")
        
        # Also show the equivalent hard erasure (single-class) for comparison
        hard_config = {
            "measurement_error_rate": Pm * (1 - sum_Rm) / 2,
            "measurement_error_rate_with_erasure": Pm * sum_Rm,
            "measurement_erasure_rate_no_error": (1 - Pm) * sum_Rc,
        }
        print(f"\n  Hard erasure (single-class) equivalent config:")
        print(f"  {json.dumps(hard_config, indent=4)}")

if __name__ == "__main__":
    main()
