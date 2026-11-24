import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, json, numpy as np
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":7.5})
S=json.load(open("/home/claude/bench/summary2.json")); A=S["ablation"]; P=S["peer"]
fig,ax=plt.subplots(1,2,figsize=(10,3.2),dpi=300,gridspec_kw={"width_ratios":[1.25,1]})
keys=["none","soil_only","trt_only","soilxtrt_raw","trt_norm"]
labs=["None\n(all plots,\nraw VWC)","Soil only\n(all plots,\navailable water)","Treatment only\n(treatment groups,\nraw VWC)","Soil × treatment\ngroups\n(raw VWC)","Treatment groups\n+ soil\nnormalisation"]
x=np.arange(5); w=0.38
v=[A[k]["f1"][0] for k in keys]; e=[A[k]["f1"][1] for k in keys]
col=["#B7B7B0","#B7B7B0","#B7B7B0","#5B8C5A","#2F6A3A"]
ax[0].bar(x-w/2,v,w,yerr=e,color=col,capsize=2,error_kw=dict(lw=.8),label="One drifting probe per group")
two=[np.nan,np.nan,np.nan,A["two_soilxtrt_raw"][0],A["two_trt_norm"][0]]
ax[0].bar(x+w/2,two,w,color="none",edgecolor=["#5B8C5A"]*4+["#2F6A3A"],hatch="////",lw=.8,label="Two drifting probes in one group")
for i,val in enumerate(v): ax[0].text(i-w/2,val+e[i]+0.025,f"{val:.3f}",ha="center",fontsize=6.3)
for i,val in enumerate(two):
    if not np.isnan(val): ax[0].text(i+w/2,val+0.025,f"{val:.3f}",ha="center",fontsize=6.3)
ax[0].set_xticks(x); ax[0].set_xticklabels(labs,fontsize=6.2); ax[0].set_ylim(0,1.08); ax[0].set_ylabel("Drift F1 (plot-days)")
ax[0].set_title("(a) Metadata supplied to the same algorithm (SaQC drift-from-norm)",fontsize=8,loc="left"); ax[0].legend(fontsize=6.3,frameon=False,loc="upper left")
sch=[("all plots (24)","All plots\n(24)"),("random (4)","Random\n(4)"),("block, mixed treatments (6)","Block,\nmixed trt (6)"),("treatment (8)","Treatment\n(8)"),("treatment x cultivar (4)","Treatment ×\ncultivar (4)"),("soil x treatment (4)","Soil ×\ntreatment (4)")]
nk={"all plots (24)":"soil_only","random (4)":"random_norm","block, mixed treatments (6)":"block_norm","treatment (8)":"trt_norm","treatment x cultivar (4)":"trtcv_norm","soil x treatment (4)":"soilxtrt_norm"}
x=np.arange(len(sch))
ax[1].bar(x-w/2,[A[nk[s]]["f1"][0] for s,_ in sch],w,yerr=[A[nk[s]]["f1"][1] for s,_ in sch],color="#2F6A3A",capsize=2,error_kw=dict(lw=.8),label="SaQC DFN on available water")
ax[1].bar(x+w/2,[P[s][0] for s,_ in sch],w,yerr=[P[s][1] for s,_ in sch],color="#C08A22",capsize=2,error_kw=dict(lw=.8),label="Peer-trend rule on raw VWC")
ax[1].set_xticks(x); ax[1].set_xticklabels([l for _,l in sch],fontsize=6.2); ax[1].set_ylim(0,1.08); ax[1].set_ylabel("Drift F1 (plot-days)")
ax[1].set_title("(b) Peer-group construction (group size)",fontsize=8,loc="left"); ax[1].legend(fontsize=6.3,frameon=False,loc="upper left")
for a in ax: a.spines[["top","right"]].set_visible(False); a.tick_params(length=2)
plt.tight_layout(); plt.savefig("fig4_metadata.png",bbox_inches="tight",facecolor="white"); print("ok")
