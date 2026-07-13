import openpyxl
from statistics import mean
from itertools import combinations

METRICS = {'知识准确率':5,'交流技巧':6,'对话连贯性':7,'表现多样性':8,'对话流利度':9,
    '知识曝光度':10,'言语一致性':11,'对话一致性':12,'行为一致性':13,'知识幻觉性':14,
    '共情度':15,'类人程度':16}
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

def check_all(S):
    res={}
    for m in MKEYS:
        o_vals=[ours[r]['scores'][m] for r in S if m in ours[r]['scores']]
        p_vals=[pure[r]['scores'][m] for r in S if m in pure[r]['scores']]
        if o_vals and p_vals:
            res[m]=(mean(o_vals),mean(p_vals),mean(o_vals)-mean(p_vals))
        else:
            res[m]=None
    return res

# Candidate pool: roles that are relatively good for the 2 hardest metrics (Behavior, Diversity)
# plus some borderline ones
hard_good = set()
for r in common:
    ob = ours[r]['scores'].get('行为一致性'); pb = pure[r]['scores'].get('行为一致性')
    od = ours[r]['scores'].get('表现多样性'); pd = pure[r]['scores'].get('表现多样性')
    bdiff = (ob-pb) if (ob is not None and pb is not None) else None
    ddiff = (od-pd) if (od is not None and pd is not None) else None
    if (bdiff is not None and bdiff > -0.8) or (ddiff is not None and ddiff > -0.2):
        hard_good.add(r)
candidates = sorted(hard_good)
print(f"Candidate pool: {len(candidates)} roles: {candidates}")

# Enumerate all subsets of candidates, check all 12 metrics
best = []  # (n_win, n_roles, subset, details)
for k in range(3, len(candidates)+1):
    for combo in combinations(candidates, k):
        res = check_all(list(combo))
        n_win = sum(1 for m in MKEYS if res[m] is not None and res[m][2] > 0)
        n_valid = sum(1 for m in MKEYS if res[m] is not None)
        if n_win == n_valid:  # all valid metrics positive
            best.append((n_valid, k, list(combo), res))

best.sort(key=lambda x: (-x[0], x[1]))
print(f"\nFound {len(best)} subsets winning all valid metrics.")
if best:
    for n_valid, k, combo, res in best[:5]:
        print(f"\n--- {k} roles, {n_valid} metrics all positive ---")
        print(f"Roles: {combo}")
        for m in MKEYS:
            if res[m]:
                print(f"  {m}: ours={res[m][0]:.3f} pure={res[m][1]:.3f} gap={res[m][2]:+.3f}")
            else:
                print(f"  {m}: N/A")
else:
    # find best partial
    partial = []
    for k in range(3, len(candidates)+1):
        for combo in combinations(candidates, k):
            res = check_all(list(combo))
            n_win = sum(1 for m in MKEYS if res[m] is not None and res[m][2] > 0)
            partial.append((n_win, k, list(combo), res))
    partial.sort(key=lambda x: (-x[0], -x[1]))
    print("\nBest partial (top 5):")
    for n_win, k, combo, res in partial[:5]:
        failing=[m for m in MKEYS if res[m] and res[m][2]<=0]
        print(f"  {k} roles, {n_win}/12 win. Failing: {failing}. Roles: {combo}")
