import json
import numpy as np

# Re=0 데이터 로드
with open('results_re0_th.json', 'r') as f:
    raw = json.load(f)
    data = raw['results']

print("=" * 60)
print("실제 데이터 포인트에서 직접 crossing 찾기 (보간 없이)")
print("=" * 60)

# 모든 code distance 쌍에 대해 raw 데이터에서 직접 crossing 찾기
for d1, d2 in [(5,7), (7,9), (9,11)]:
    p1 = np.array(data[str(d1)]['p'])
    pL1 = np.array(data[str(d1)]['pL'])
    p2 = np.array(data[str(d2)]['p'])
    pL2 = np.array(data[str(d2)]['pL'])
    
    # 공통 p
    common_p = sorted(set(p1) & set(p2))
    print(f'\nd={d1} vs d={d2}:')
    
    prev_diff = None
    prev_common_p = None
    for p in common_p:
        idx1 = np.where(np.isclose(p1, p))[0][0]
        idx2 = np.where(np.isclose(p2, p))[0][0]
        if pL1[idx1] > 0 and pL2[idx2] > 0:
            diff = np.log(pL1[idx1]) - np.log(pL2[idx2])
            print(f"  p={p:.6f}: pL({d1})={pL1[idx1]:.6f}, pL({d2})={pL2[idx2]:.6f}, diff={diff:.4f}")
            if prev_diff is not None and prev_diff > 0 and diff < 0:
                # 선형 보간으로 crossing 찾기
                cross_p = prev_common_p + (p - prev_common_p) * prev_diff / (prev_diff - diff)
                print(f"  >>> CROSSING between p={prev_common_p:.5f} and p={p:.5f}")
                print(f"  >>> p_cross = {cross_p:.6f} ({cross_p*100:.3f}%)")
            prev_diff = diff
            prev_common_p = p
