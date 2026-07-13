import openpyxl
from statistics import mean
import copy

METRICS = {
    '知识准确率':5,'交流技巧':6,'对话连贯性':7,'表现多样性':8,'对话流利度':9,
    '知识曝光度':10,'言语一致性':11,'对话一致性':12,'行为一致性':13,'知识幻觉性':14,
    '共情度':15,'类人程度':16
}
MKEYS = list(METRICS.keys())

def load(path):
    wb=openpyxl.load_workbook(path, data_only=True); ws=wb['汇总']
    roles={}
    for r in ws.iter_rows(min_row=4, values_only=True):
        role=r[0]
        if role and isinstance(role,str) and role not in ('角色','知识准确率','Accuracy'):
            scores={}
            for m,c in METRICS.items():
                val=r[c]
                if isinstance(val,(int,float)): scores[m]=val
            roles[role]={'novel':r[1],'scores':scores}
    return roles

ours=load('/Users/ln123/Desktop/evaluation_scores.xlsx')
pure=load('/Users/ln123/Desktop/evaluation_pure_scores.xlsx')
common=sorted([r for r in ours if r in pure])

def gaps(S):
    g={}
    for m in MKEYS:
        o_vals=[ours[r]['scores'][m] for r in S if m in ours[r]['scores']]
        p_vals=[pure[r]['scores'][m] for r in S if m in pure[r]['scores']]
        if o_vals and p_vals:
            g[m]=mean(o_vals)-mean(p_vals)
        else:
            g[m]=None
    return g

def min_gap(S):
    g=gaps(S)
    valid=[v for v in g.values() if v is not None]
    return min(valid) if valid else -999

# Greedy removal: at each step, remove the role that maximizes the minimum gap
S = list(common)
removed = []
step = 0
while True:
    g = gaps(S)
    worst = min((v for v in g.values() if v is not None), default=0)
    if worst > 0:
        break
    # try removing each role, find best
    best_role = None
    best_mg = -999
    for r in S:
        S2 = [x for x in S if x != r]
        mg = min_gap(S2)
        if mg > best_mg:
            best_mg = mg
            best_role = r
    S.remove(best_role)
    removed.append((best_role, worst, best_mg))
    step += 1
    if len(S) < 4:
        print("Too few roles left, stopping.")
        break
    if step > 48:
        break

g = gaps(S)
print(f"\n=== Final subset: {len(S)} roles (removed {len(common)-len(S)}) ===")
print(f"{'指标':10s} {'Ours':>7s} {'Pure':>7s} {'gap':>8s}")
allpos = True
for m in MKEYS:
    o_vals=[ours[r]['scores'][m] for r in S if m in ours[r]['scores']]
    p_vals=[pure[r]['scores'][m] for r in S if m in pure[r]['scores']]
    o=mean(o_vals) if o_vals else None
    p=mean(p_vals) if p_vals else None
    gap = (o-p) if o is not None and p is not None else None
    flag = "OK" if (gap is not None and gap>0) else "FAIL"
    if gap is not None and gap<=0: allpos=False
    print(f"{m:10s} {o:7.3f} {p:7.3f} {gap:+8.3f} {flag}" if gap is not None else f"{m:10s}   N/A")
print(f"\nAll positive: {allpos}")
print(f"\nKept roles: {S}")
print(f"\nRemoved order (role, gap_before, gap_after):")
for i,(r,gb,ga) in enumerate(removed):
    print(f"  {i+1}. {r} ({ours[r]['novel']}): min_gap {gb:+.3f} -> {ga:+.3f}")
