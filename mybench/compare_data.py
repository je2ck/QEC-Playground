import json
import numpy as np

# Re=0 데이터
with open('results_re0_th.json') as f:
    re0 = json.load(f)['results']

# pm=0 데이터
try:
    with open('results_measurement_error/results_pm0.json') as f:
        pm0 = json.load(f)['results']
except:
    pm0 = None
    print('results_pm0.json not found')

def find_crossing(data, d1, d2):
    """두 code distance 간 crossing point 찾기"""
    p1 = np.array(data[str(d1)]['p'])
    pL1 = np.array(data[str(d1)]['pL'])
    p2 = np.array(data[str(d2)]['p'])
    pL2 = np.array(data[str(d2)]['pL'])
    
    # 공통 p
    common_p = sorted(set(p1) & set(p2))
    
    prev_diff = None
    prev_p = None
    for p in common_p:
        idx1 = np.where(np.isclose(p1, p))[0][0]
        idx2 = np.where(np.isclose(p2, p))[0][0]
        if pL1[idx1] > 0 and pL2[idx2] > 0:
            diff = np.log(pL1[idx1]) - np.log(pL2[idx2])
            if prev_diff is not None and prev_diff > 0 and diff < 0:
                cross_p = prev_p + (p - prev_p) * prev_diff / (prev_diff - diff)
                return cross_p
            prev_diff = diff
            prev_p = p
    return None

print('=== Re=0 데이터 crossing points ===')
for d1, d2 in [(5,7), (7,9), (9,11)]:
    cp = find_crossing(re0, d1, d2)
    if cp:
        print(f'd={d1} vs d={d2}: p_th = {cp*100:.3f}%')
    else:
        print(f'd={d1} vs d={d2}: no crossing found')

if pm0:
    print()
    print('=== pm=0 데이터 crossing points ===')
    for d1, d2 in [(5,7), (7,9), (9,11)]:
        cp = find_crossing(pm0, d1, d2)
        if cp:
            print(f'd={d1} vs d={d2}: p_th = {cp*100:.3f}%')
        else:
            print(f'd={d1} vs d={d2}: no crossing found')
