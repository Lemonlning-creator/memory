import openpyxl
from statistics import mean

# metric display name -> column index (0-based) in 汇总 sheet
# header: 0角色 1作品 2Ours 3存在指标数 4评分记录数 5知识准确率 6交流技巧 7对话连贯性
#         8表现多样性 9对话流利度 10知识曝光度 11言语一致性 12对话一致性 13行为一致性
#         14知识幻觉性 15共情度 16类人程度 17备注
METRICS = {
    '知识准确率':5,'交流技巧':6,'对话连贯性':7,'表现多样性':8,'对话流利度':9,
    '知识曝光度':10,'言语一致性':11,'对话一致性':12,'行为一致性':13,'知识幻觉性':14,
    '共情度':15,'类人程度':16
}

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

def avg(data,m,roles):
    vals=[data[r]['scores'][m] for r in roles if m in data[r]['scores']]
    return (mean(vals),len(vals)) if vals else (None,0)

print(f'Ours roles={len(ours)} Pure roles={len(pure)} Common={len(common)}')
print(f"\n{'指标':10s} {'Ours(avg,n)':>16s} {'Pure(avg,n)':>16s} {'gap':>8s}")
gaps={}
for m in METRICS:
    o,on=avg(ours,m,common); p,pn=avg(pure,m,common)
    g=(o-p) if (o is not None and p is not None) else None
    gaps[m]=g
    print(f"{m:10s} {o:7.3f}({on:2d})   {p:7.3f}({pn:2d})   {g:+8.3f}" if g is not None else f"{m:10s}  N/A")
behind=[m for m in METRICS if gaps[m] is not None and gaps[m]<0]
print(f"\n落后指标({len(behind)}): {[m+str(round(gaps[m],3)) for m in behind]}")

print("\n\n=== 行为一致性 (Behavior) per-role (both have it) ===")
pairs=[]
for r in common:
    if '行为一致性' in ours[r]['scores'] and '行为一致性' in pure[r]['scores']:
        pairs.append((r, ours[r]['scores']['行为一致性'], pure[r]['scores']['行为一致性']))
print(f"{'role':10s} {'Ours':>7s} {'Pure':>7s} {'diff':>7s}")
for r,o,p in sorted(pairs,key=lambda x:x[1]-x[2]):
    print(f"{r:10s} {o:7.3f} {p:7.3f} {o-p:+7.3f}")
os=[x[1] for x in pairs]; ps=[x[2] for x in pairs]
print(f"ours range: {min(os):.2f}-{max(os):.2f}, pure range: {min(ps):.2f}-{max(ps):.2f}")
print(f"roles where ours>pure: {sum(1 for x in pairs if x[1]>x[2])}/{len(pairs)}")

print("\n\n=== 表现多样性 (Diversity) per-role (both have it) ===")
pairs=[]
for r in common:
    if '表现多样性' in ours[r]['scores'] and '表现多样性' in pure[r]['scores']:
        pairs.append((r, ours[r]['scores']['表现多样性'], pure[r]['scores']['表现多样性']))
print(f"{'role':10s} {'Ours':>7s} {'Pure':>7s} {'diff':>7s}")
for r,o,p in sorted(pairs,key=lambda x:x[1]-x[2]):
    print(f"{r:10s} {o:7.3f} {p:7.3f} {o-p:+7.3f}")
os=[x[1] for x in pairs]; ps=[x[2] for x in pairs]
print(f"ours range: {min(os):.2f}-{max(os):.2f}, pure range: {min(ps):.2f}-{max(ps):.2f}")
print(f"roles where ours>pure: {sum(1 for x in pairs if x[1]>x[2])}/{len(pairs)}")
