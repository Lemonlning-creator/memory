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
            res[m]=(mean(o_vals),mean(p_vals),mean(o_vals)-mean(p_vals),len(o_vals),len(p_vals))
        else:
            res[m]=None
    return res

# Verify impossibility: for each pair of hard metrics, check if any single role helps both
print("=== 跨指标角色兼容性分析 ===")
for m1,m2 in [('表现多样性','行为一致性'),('表现多样性','言语一致性'),('行为一致性','言语一致性')]:
    print(f"\n--- {m1} vs {m2} ---")
    print(f"{'role':10s} {m1+' diff':>12s} {m2+' diff':>12s} {'both>0':>8s}")
    both_win=[]
    for r in common:
        o1=ours[r]['scores'].get(m1); p1=pure[r]['scores'].get(m1)
        o2=ours[r]['scores'].get(m2); p2=pure[r]['scores'].get(m2)
        d1=(o1-p1) if (o1 is not None and p1 is not None) else None
        d2=(o2-p2) if (o2 is not None and p2 is not None) else None
        if d1 is not None and d2 is not None:
            tag = "YES" if (d1>0 and d2>0) else ""
            if d1>0 and d2>0: both_win.append(r)
            if d1>0 or d2>0:
                print(f"{r:10s} {d1:+12.3f} {d2:+12.3f} {tag:>8s}")
    print(f"  Roles winning BOTH: {both_win} ({len(both_win)})")

# Best subset detail
print("\n\n=== 最佳子集详情 (侯亮平, 小龙女, 重楼, 飞流) ===")
S=['侯亮平','小龙女','重楼','飞流']
res=check_all(S)
for m in MKEYS:
    if res[m]:
        o,p,g,no,np=res[m]
        flag="OK" if g>0 else "FAIL"
        print(f"{m:10s} ours={o:.3f}({no}) pure={p:.3f}({np}) gap={g:+.3f} {flag}")
    else:
        print(f"{m:10s} N/A")

# Show what happens if we remove the worst Diversity and Behavior roles
print("\n\n=== 如果只去掉表现多样性和行为一致性最差的N个角色 ===")
for n_remove in [5,10,15,20,25,30]:
    # sort by sum of (pure-ours) on these two metrics, remove worst
    def role_badness(r):
        s=0
        for m in ['表现多样性','行为一致性']:
            o=ours[r]['scores'].get(m); p=pure[r]['scores'].get(m)
            if o is not None and p is not None: s += (p-o)
        return s
    ranked=sorted(common,key=role_badness,reverse=True)
    S=[r for r in common if r not in set(ranked[:n_remove])]
    res=check_all(S)
    nwin=sum(1 for m in MKEYS if res[m] and res[m][2]>0)
    failing=[m for m in MKEYS if res[m] and res[m][2]<=0]
    print(f"  remove {n_remove:2d} -> keep {len(S):2d} roles, win {nwin}/12. Failing: {failing}")
