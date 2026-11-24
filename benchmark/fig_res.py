import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5})
t=pd.read_csv("/home/claude/bench/test_results.csv")
fig,ax=plt.subplots(1,3,figsize=(10,3.0),dpi=300,gridspec_kw={"width_ratios":[1,1.35,1.35]})
sp=["Range check","Hampel filter","SaQC (range + offset)","Proposed"]
lab_sp=["Range\ncheck","Hampel\nfilter","SaQC\n(range+offset)","Proposed"]
col_sp=["#B7B7B0","#8DA9C4","#5B8C5A","#C08A22"]
g=t[t.task=="spike"].groupby("method").f1
ax[0].bar(range(4),[g.mean()[m] for m in sp],yerr=[g.std()[m] for m in sp],color=col_sp,capsize=2,width=.65,error_kw=dict(lw=.8))
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(lab_sp,fontsize=6.5); ax[0].set_ylim(0,1.05); ax[0].set_ylabel("F1 (hourly readings)"); ax[0].set_title("(a) Spike detection",fontsize=8,loc="left")
dr=["Single-series trend","SaQC drift-from-norm, all plots","Peer trend, no gating","Proposed","SaQC drift-from-norm"]
lab_dr=["Single-\nseries\ntrend","SaQC DFN\nall plots\n(no trial info)","Peer trend\nno gating","Proposed\n(peer trend\n+ gating)","SaQC DFN\ntrial-aware\ngroups"]
col_dr=["#B7B7B0","#B7B7B0","#8DA9C4","#C08A22","#5B8C5A"]
d=t[t.task=="drift"].groupby("method")
ax[1].bar(range(5),[d.day_f1.mean()[m] for m in dr],yerr=[d.day_f1.std()[m] for m in dr],color=col_dr,capsize=2,width=.65,error_kw=dict(lw=.8))
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(lab_dr,fontsize=6.2); ax[1].set_ylim(0,1.05); ax[1].set_ylabel("F1 (plot-days)"); ax[1].set_title("(b) Drift detection",fontsize=8,loc="left")
ax[2].bar(range(5),[d.false_alarms.mean()[m] for m in dr],yerr=[d.false_alarms.std()[m] for m in dr],color=col_dr,capsize=2,width=.65,error_kw=dict(lw=.8))
ax[2].set_xticks(range(5)); ax[2].set_xticklabels(lab_dr,fontsize=6.2); ax[2].set_yscale("symlog",linthresh=1); ax[2].set_ylim(0,150)
ax[2].set_yticks([0,1,2,5,10,20,50,100]); ax[2].set_yticklabels(["0","1","2","5","10","20","50","100"])
ax[2].set_ylabel("False-alarm episodes per season"); ax[2].set_title("(c) Drift false alarms",fontsize=8,loc="left")
for a in ax:
    a.spines[["top","right"]].set_visible(False); a.tick_params(length=2)
plt.tight_layout(); plt.savefig("fig3_results.png",bbox_inches="tight",facecolor="white"); print("ok")
