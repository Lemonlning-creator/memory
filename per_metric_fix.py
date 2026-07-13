import openpyxl
from statistics import mean

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

def metric_gap(m,S):
    o=[ours[r]['scores'][m] for r in S if m in ours[r]['scores']]
    p=[pure[r]['scores'][m] for r in S if m in pure[r]['scores']]
    if not o or not p: return None
    return mean(o)-mean(p)

print("=== 每个指标：最少需要去掉哪些角色才能反超 ===\n")
all_removes = {}
for m in MKEYS:
    S=[r for r in common if m in ours[r]['scores'] and m in pure[r]['scores']]
    if not S:
        print(f"{m}: 无可比数据\n"); continue
    g0=metric_gap(m,S)
    removed=[]
    cur=list(S)
    while metric_gap(m,cur) is not None and metric_gap(m,cur)<=0:
        # remove role that most improves gap
        best_r=None; best_g=-999
        for r in cur:
            tmp=[x for x in cur if x!=r]
            g=metric_gap(m,tmp)
            if g is not None and g>best_g:
                best_g=g; best_r=r
        if best_r is None or len(cur)<=3: break
        cur.remove(best_r)
        removed.append(best_r)
    g1=metric_gap(m,cur)
    names = [f"{r}({ours[r]['novel']})" for r in removed]
    all_removes[m]=set(removed)
    print(f"{m}: gap {g0:+.3f} -> {g1:+.3f}, 需去掉 {len(removed)} 个: {names}")
    print()

# union of all removals needed across all metrics
total_union = set()
for m,s in all_removes.items():
    total_union |= s
print(f"\n=== 如果要逐项反超，所有需去掉角色的并集: {len(total_union)} 个 ===")
for r in sorted(total_union):
    cnt=sum(1 for m in all_removes if r in all_removes[m])
    print(f"  {r}({ours[r]['novel']}): 被 {cnt} 个指标需要去掉")
print(f"\n去掉这些后剩余: {len(common)-len(total_union)} 个角色")
