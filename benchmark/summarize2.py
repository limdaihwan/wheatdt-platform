import pandas as pd, json, numpy as np
t=pd.read_csv("test_results.csv"); g=pd.read_csv("grouping_results.csv"); gn=pd.read_csv("grouping_normalized.csv")
rob=pd.read_csv("robustness_results.csv"); rn=pd.read_csv("rob_normalized_trt8.csv"); pl=pd.read_csv("plausibility.csv")
def ms(df,col): return [round(float(df[col].mean()),3), round(float(df[col].std()),3)]
def sel(df,**k):
    m=np.ones(len(df),bool)
    for a,b in k.items(): m&=(df[a]==b).values
    return df[m]
A={}
d0=sel(t,task="drift",method="SaQC drift-from-norm, all plots"); A["none"]=dict(f1=ms(d0,"day_f1"),er=ms(d0,"event_recall"),fa=ms(d0,"false_alarms"))
for key,df in [("soil_only",sel(gn,scheme="all plots (24)")),("trt_only",sel(g,scheme="treatment (8)",method="SaQC DFN")),
               ("soilxtrt_raw",sel(g,scheme="soil x treatment (4)",method="SaQC DFN")),("trt_norm",sel(gn,scheme="treatment (8)")),
               ("block_norm",sel(gn,scheme="block, mixed treatments (6)")),("random_norm",sel(gn,scheme="random (4)")),
               ("trtcv_norm",sel(gn,scheme="treatment x cultivar (4)")),("soilxtrt_norm",sel(gn,scheme="soil x treatment (4)"))]:
    A[key]=dict(f1=ms(df,"day_f1"),er=ms(df,"event_recall"),fa=ms(df,"false_alarms"))
A["two_soilxtrt_raw"]=ms(sel(rob,scenario="two drifts in one group",method="SaQC DFN, trial groups"),"day_f1")
A["two_trt_norm"]=ms(sel(rn,scenario="two drifts in one group"),"day_f1")
A["two_peer"]=ms(sel(rob,scenario="two drifts in one group",method="Proposed peer trend + gate"),"day_f1")
peer={s:ms(sel(g,scheme=s,method="Peer trend + gate"),"day_f1") for s in g.scheme.unique()}
f=lambda df,v,r=3:{k:{kk:round(float(vv),r) for kk,vv in d.items()} for k,d in df.pivot_table(index="method",columns="scenario",values=v,aggfunc="mean").to_dict().items()}
robd=f(rob[rob.task=="drift"],"day_f1"); robs=f(rob[rob.task=="spike"],"f1"); robfa=f(rob[rob.task=="drift"],"false_alarms",1)
plaus={c:[round(float(pl[c].mean()),2),round(float(pl[c].std()),2),round(float(pl[c].min()),2),round(float(pl[c].max()),2)] for c in pl.columns if c!="seed"}
json.dump(dict(ablation=A,peer=peer,rob_drift=robd,rob_spike=robs,rob_fa=robfa,plaus=plaus),open("summary2.json","w"),indent=1)
for k,v in A.items(): print(k,v)
print(peer)
