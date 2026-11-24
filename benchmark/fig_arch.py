import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams["font.family"]="DejaVu Sans"
fig,ax=plt.subplots(figsize=(10,4.3),dpi=300); ax.set_xlim(0,100); ax.set_ylim(0,38); ax.axis("off")
def box(x,y,w,h,title,lines,fc,ec="#33413A"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",fc=fc,ec=ec,lw=1))
    ax.text(x+w/2,y+h-2.3,title,ha="center",va="top",fontsize=8.6,weight="bold",color="#1B2620")
    ax.text(x+w/2,y+h-6.2,"\n".join(lines),ha="center",va="top",fontsize=6.9,color="#2A3530",linespacing=1.35)
def arr(x1,y1,x2,y2,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=9,lw=1,color="#4A5750",linestyle=ls))
Y=13.5; H=19
box(1,Y,16,H,"Data sources",["Weather station","(T, RH, VPD, rain, Rs, ET₀)","Soil probes 10/30/60 cm","UAV NDVI / NDRE / CTD","Phenology, biomass,","harvest, management"],"#E3ECF3")
box(21,Y,16,H,"Ingestion",["8-table schema","(plots, weather, soil,","canopy, phenology,","biomass, harvest,","management)","ISO 8601, fixed units"],"#EEF1EA")
box(41,Y,18,H,"Trial-aware QC",["Range + spike filter","Peer-residual drift test","with peer-stability gate","Outage detection","NDVI dip rejection","(flags kept, not deleted)"],"#F6EBD3")
box(63,Y,17,H,"Physiological layer",["Root-zone available","water (RAW)","Canopy–air temp. (CTD)","Thermal time, Zadoks","stage interpolation","Rule-based alerts"],"#E6F0E1")
box(84,Y,15,H,"Dashboard",["Field map (24 plots)","Season timeline","Plot soil/canopy","Weather, yield","QC log"],"#EDE6F2")
for x1,x2 in [(17.8,20.2),(37.8,40.2),(59.8,62.2),(80.8,83.2)]: arr(x1,Y+H/2,x2,Y+H/2)
box(21,0.6,38,9.2,"",[],"#FFFFFF","#8A948E")
ax.text(40,8.4,"Synthetic data generator (this study)",ha="center",va="top",fontsize=8.2,weight="bold")
ax.text(40,4.9,"water-balance-driven crop–soil model, block and plot random effects,\nlabelled fault injection (spikes, drift, outages) → ground truth for benchmarking",ha="center",va="top",fontsize=6.7,color="#2A3530")
arr(30,10.4,30,13.3,ls="--"); arr(50,10.4,50,13.3,ls="--")
ax.text(52,11.9,"truth labels",fontsize=6.3,color="#4A5750",va="center")
ax.text(1,36,"Field trial: 3 irrigation levels × 2 cultivars × 4 blocks = 24 plots",fontsize=7.6,color="#4A5750")
plt.savefig("fig1a_arch.png",bbox_inches="tight",facecolor="white"); print("ok")
