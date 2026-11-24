import sys, json, bench as B
mode=sys.argv[1]; seeds=[int(x) for x in sys.argv[2:]]
if mode=="tune":
    df=B.run(seeds)
else:
    tp=json.load(open(f"{B.BENCH}/tuned_params.json"))
    df=B.run(seeds, tp["spike"], tp["drift"])
df.to_csv(f"{B.BENCH}/part_{mode}_{'_'.join(map(str,seeds))}.csv", index=False)
