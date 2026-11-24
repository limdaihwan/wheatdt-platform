"""
밀 디지털트윈 플랫폼용 가상(synthetic) 데이터 생성기  v2
-----------------------------------------------------------
원본 코드의 구조(8개 테이블, 스키마, 관리 일정)를 유지하되 다음을 수정/보완함.

[버그 수정]
 1) 기상 종료일 2024-06-30 -> 2025-07-15 (원본은 시간 수가 음수가 되어 기상 테이블이 비고 전체가 중단됨)
 2) 계절 기온 곡선 위상 오류 수정 (원본: 11월 평균 -9°C, 4월 21°C -> 수정: 바야돌리드 평년 수준)
 3) canopy.csv의 source 컬럼: 원본은 절반을 'measured'로 표기 -> 가상데이터이므로 전부 'synthetic'

[생리적 인과 구조 보완]
 기상(ET0, 강우) -> 관개 -> 토양수분(물수지) -> 수분스트레스 계수 Ks
   -> 피복도/LAI/NDVI, CTD, 초장, 바이오매스(RUE), 수량/천립중/단백질
 처리 간 차이는 처리 라벨로 직접 부여하지 않고, 관개량 차이 -> 토양수분 -> Ks 경로로 발생.

[통계 구조]
 반복(block) 효과 + 포장별 랜덤효과 + 측정오차, 결측/이상치/센서 드리프트/통신두절 주입.
 정답값(ground truth)은 _truth/ 폴더에 별도 저장 -> 혼합모형·전처리 모듈 검증용 (모델 입력 금지)
"""
import os
import numpy as np
import pandas as pd

SEED = int(os.environ.get("SEED", 42))
FAULTS = os.environ.get("FAULTS", "1") == "1"
OUT = os.environ.get("OUT_DIR", "data")
TRUTH = os.path.join(OUT, "_truth")
os.makedirs(TRUTH, exist_ok=True)
rng = np.random.default_rng(SEED)

LAT = 41.65  # 바야돌리드 인근
SEASON = "2024-25"
SOWING = pd.Timestamp("2024-11-15")
W_START = pd.Timestamp("2024-11-01 00:00")
W_END = pd.Timestamp("2025-07-15 23:00")

# ==========================================================
# 1. 포장 메타데이터 (plots.csv)
# ==========================================================
blocks = [1, 2, 3, 4]
irr_trts = ["FULL", "DEFICIT", "RAINFED"]
cultivars = ["CV_A", "CV_B"]
soil_hyd = {"clay loam": dict(fc=36.0, wp=18.0), "loam": dict(fc=30.0, wp=12.0)}
block_mult = {1: 1.04, 2: 1.015, 3: 0.98, 4: 0.965}  # 반복(블록) 효과 (생산성 배수)

plots, truth_plot = [], []
pidx = 1
for b in blocks:
    for irr in irr_trts:
        for cv in cultivars:
            pid = f"P{pidx:02d}"
            soil = "clay loam" if b in (1, 2) else "loam"
            plot_re = float(rng.normal(0, 0.035))  # 포장 랜덤효과
            plots.append(dict(
                plot_id=pid, block=b, irrigation_trt=irr, cultivar=cv,
                plot_area=round(float(rng.uniform(12, 18)), 1),
                lat=round(LAT + b * 0.005 + rng.uniform(-0.001, 0.001), 5),
                lon=round(-4.72 + pidx * 0.003 + rng.uniform(-0.001, 0.001), 5),
                soil_texture=soil, sowing_date=SOWING.strftime("%Y-%m-%d"), season=SEASON))
            truth_plot.append(dict(plot_id=pid, block_mult=block_mult[b], plot_re=round(plot_re, 4),
                                   prod_mult=round(block_mult[b] * (1 + plot_re), 4)))
            pidx += 1
df_plots = pd.DataFrame(plots)
df_truth = pd.DataFrame(truth_plot)
df_plots.to_csv(f"{OUT}/plots.csv", index=False)

# ==========================================================
# 2. 기상 (weather.csv) - 시간별
# ==========================================================
def ra_fao56(doy, lat_deg):
    phi = np.radians(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(dec))
    return 24 * 60 / np.pi * 0.082 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))

rain_prob = {11: 0.30, 12: 0.28, 1: 0.27, 2: 0.25, 3: 0.24, 4: 0.26, 5: 0.20, 6: 0.10, 7: 0.06}
wx_rows, daily_wx = [], []
anom = 0.0
for day in pd.date_range(W_START.normalize(), W_END.normalize(), freq="D"):
    doy = day.dayofyear
    anom = 0.7 * anom + rng.normal(0, 1.8)  # 날씨 지속성(AR1)
    is_rain = rng.random() < rain_prob[day.month]
    rain_mm = float(rng.gamma(0.8, 6.0)) if is_rain else 0.0
    tmean = 12.5 - 8.5 * np.cos(2 * np.pi * (doy - 15) / 365) + anom - (1.0 if is_rain else 0)
    dtr = (12.5 + 3.5 * np.cos(2 * np.pi * (doy - 196) / 365)) * (0.55 if is_rain else 1.0)
    rh_mean = 68 + 16 * np.cos(2 * np.pi * (doy - 15) / 365) + (12 if is_rain else 0)
    daylen = 12.3 + 2.9 * np.cos(2 * np.pi * (doy - 172) / 365)
    rise = 13.5 - daylen / 2
    peak = 450 + 500 * (0.5 + 0.5 * np.cos(2 * np.pi * (doy - 172) / 365))
    cloud = rng.uniform(0.3, 0.6) if is_rain else rng.uniform(0.8, 1.0)
    rain_h = np.zeros(24)
    if rain_mm > 0:
        st, dur = rng.integers(0, 20), rng.integers(3, 9)
        hrs = np.arange(st, min(24, st + dur))
        rain_h[hrs] = rain_mm * rng.dirichlet(np.ones(len(hrs)))
    temps = []
    for h in range(24):
        t = tmean + dtr / 2 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 0.4)
        rh = float(np.clip(rh_mean - 2.8 * (t - tmean) + rng.normal(0, 3), 15, 100))
        es = 0.61078 * np.exp(17.27 * t / (t + 237.3))
        x = (h + 0.5 - rise) / daylen
        rad = peak * cloud * np.sin(np.pi * x) if 0 < x < 1 else 0.0
        wx_rows.append(dict(timestamp=day + pd.Timedelta(hours=h), t_air=round(t, 1), rh=round(rh, 1),
                            vpd=round(max(0.0, es * (1 - rh / 100)), 3), rainfall=round(rain_h[h], 1),
                            solar_rad=round(max(0.0, rad), 1), wind_speed=round(float(rng.weibull(2.0) * 2.5), 1)))
        temps.append(t)
    tmax, tmin, tm = max(temps), min(temps), float(np.mean(temps))
    et0 = 0.0023 * (tm + 17.8) * np.sqrt(max(tmax - tmin, 0.1)) * ra_fao56(doy, LAT) * 0.408
    daily_wx.append(dict(date=day, tmean=tm, et0=round(float(np.clip(et0, 0.3, 8.0)), 2)))

df_w = pd.DataFrame(wx_rows)
df_wd = pd.DataFrame(daily_wx).set_index("date")
df_w["et0"] = df_w["timestamp"].dt.normalize().map(df_wd["et0"])
# 일사 합계(MJ/m2/day) - 내부 계산용
df_wd["rs_mj"] = df_w.groupby(df_w["timestamp"].dt.normalize())["solar_rad"].sum() * 0.0036
df_wd["rain"] = df_w.groupby(df_w["timestamp"].dt.normalize())["rainfall"].sum()

df_w_out = df_w.copy()
miss = rng.random(len(df_w_out)) < 0.003  # 기상 결측 0.3%
df_w_out.loc[miss, ["t_air", "rh", "vpd"]] = np.nan
df_w_out["timestamp"] = df_w_out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M")
df_w_out.to_csv(f"{OUT}/weather.csv", index=False)

# ==========================================================
# 8. 관리 이벤트 (management.csv) - 원본 일정 유지
# ==========================================================
mg = []
for _, p in df_plots.iterrows():
    pid, trt = p.plot_id, p.irrigation_trt
    mg += [dict(plot_id=pid, date="2024-11-15", event_type="FERTILIZER", amount=50, product="NPK 15-15-15"),
           dict(plot_id=pid, date="2025-02-25", event_type="FERTILIZER", amount=60, product="Urea 46% N"),
           dict(plot_id=pid, date="2025-03-10", event_type="PESTICIDE", amount=2.5, product="Tribenuron-methyl")]
    sched = {"FULL": (["2025-04-05", "2025-04-20", "2025-05-05", "2025-05-20", "2025-06-02"], 35),
             "DEFICIT": (["2025-04-15", "2025-05-15"], 25), "RAINFED": ([], 0)}[trt]
    mg += [dict(plot_id=pid, date=d, event_type="IRRIGATION", amount=sched[1], product="Water") for d in sched[0]]
df_mg = pd.DataFrame(mg)
df_mg.to_csv(f"{OUT}/management.csv", index=False)
irr_map = {(r.plot_id, pd.Timestamp(r.date)): r.amount for r in df_mg[df_mg.event_type == "IRRIGATION"].itertuples()}
fert_dates = [pd.Timestamp("2024-11-15"), pd.Timestamp("2025-02-25")]

# ==========================================================
# 5. 생육단계 (phenology.csv) - GDD(기저 0°C) 기반
# ==========================================================
gdd_daily = df_wd.loc[df_wd.index >= SOWING, "tmean"].clip(lower=0)
gdd_cum = gdd_daily.cumsum()
targets = {"CV_A": {10: 150, 30: 600, 65: 1300, 92: 1900},
           "CV_B": {10: 140, 30: 560, 65: 1240, 92: 1830}}
ph_rows, stage_gdd = [], {}
for _, p in df_plots.iterrows():
    sg = {}
    for st, tg in targets[p.cultivar].items():
        adj = tg + rng.integers(-15, 16)
        if st == 92:
            adj -= {"FULL": 0, "DEFICIT": 25, "RAINFED": 60}[p.irrigation_trt]  # 가뭄 시 등숙기간 단축
        hit = gdd_cum[gdd_cum >= adj]
        if hit.empty:
            raise RuntimeError(f"{p.plot_id} stage {st}: GDD {adj} 미도달 - 기상 기간 연장 필요")
        sg[st] = adj
        ph_rows.append(dict(plot_id=p.plot_id, date=hit.index[0].strftime("%Y-%m-%d"),
                            zadoks_stage=st, gdd_cum=round(float(hit.iloc[0]), 1)))
    stage_gdd[p.plot_id] = sg
df_ph = pd.DataFrame(ph_rows)
df_ph.to_csv(f"{OUT}/phenology.csv", index=False)

# ==========================================================
# 3·4·6 공통: 포장별 일/시간 단위 작물-토양 시뮬레이션
# ==========================================================
def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

THICK = np.array([150.0, 200.0, 350.0])  # 10cm / 30cm / 60cm 센서가 대표하는 층 두께(mm)
days_all = df_wd.index
hour_solar = df_w.set_index("timestamp")["solar_rad"]
hour_rain = df_w.set_index("timestamp")["rainfall"]
hour_tair = df_w.set_index("timestamp")["t_air"]

soil_rows, daily_state = [], []
for _, p in df_plots.iterrows():
    pid, cv = p.plot_id, p.cultivar
    hyd = soil_hyd[p.soil_texture]
    fc_v, wp_v = hyd["fc"], hyd["wp"]
    pm = float(df_truth.loc[df_truth.plot_id == pid, "prod_mult"].iloc[0])
    sg = stage_gdd[pid]
    cover_max = min(0.96, (0.95 if cv == "CV_A" else 0.92) * pm)
    h_max = (88.0 if cv == "CV_A" else 80.0) * pm
    theta = np.array([fc_v * 0.80, fc_v * 0.90, fc_v * 0.95])  # 초기 체적수분(%)
    soil_t = 10.0
    ks_hist, W, H, cover_post, s_prev = [], 0.0, 5.0, None, 0.0
    ks = 1.0
    for day in days_all:
        g = float(gdd_cum.get(day, 0.0)) if day >= SOWING else 0.0
        # --- 잠재 피복도 (생육단계 기반) ---
        if g < sg[10]:
            cov_pot = 0.0
        elif g < sg[65]:
            cov_pot = cover_max * smooth(((g - sg[10]) / (sg[65] - sg[10])) ** 0.7)
        else:
            sen0 = sg[65] + 0.35 * (sg[92] - sg[65])
            cov_pot = cover_max * (1 - smooth((g - sen0) / (sg[92] - sen0))) if g > sen0 else cover_max
            cov_pot = max(cov_pot, 0.03)
        ks14 = float(np.mean(ks_hist[-14:])) if ks_hist else 1.0
        cov = cov_pot * (0.5 + 0.5 * ks14)
        if g >= sg[65]:  # 개화 후 재녹화 없음
            cover_post = cov if cover_post is None else min(cover_post, cov)
            cov = cover_post
        lai = min(6.0, -np.log(1 - min(cov, 0.96)) / 0.55)
        # --- 증발산 수요 ---
        kc = 0.3 + 0.85 * cov
        et_day = kc * df_wd.at[day, "et0"] * ks
        root_w = np.array([0.6, 0.3, 0.1]) * (1 - cov) + np.array([0.35, 0.35, 0.30]) * cov
        hrs = pd.date_range(day, periods=24, freq="h")
        sol = hour_solar.loc[hrs].values
        sol_frac = sol / sol.sum() if sol.sum() > 0 else np.full(24, 1 / 24)
        irr = irr_map.get((pid, day), 0.0)
        for i, ts in enumerate(hrs):
            water = hour_rain.at[ts] + (irr if i == 8 else 0.0)  # 관개는 08시
            for L in range(3):  # 침투: 포화(fc+6) 초과분은 하층으로
                cap = (fc_v + 6 - theta[L]) / 100 * THICK[L]
                add = min(water, max(cap, 0))
                theta[L] += add / THICK[L] * 100
                water -= add
            for L in range(3):  # 중력배수: FC 초과분의 8%/h
                ex = max(theta[L] - fc_v, 0) * 0.08
                theta[L] -= ex
                if L < 2:
                    theta[L + 1] += ex * THICK[L] / THICK[L + 1]
            avail = np.clip((theta - wp_v) / (fc_v - wp_v), 0, 1)
            w = root_w * avail
            w = w / w.sum() if w.sum() > 0 else root_w
            theta -= et_day * sol_frac[i] * w / THICK * 100
            theta = np.maximum(theta, wp_v - 4)
            soil_t += 0.06 * (hour_tair.at[ts] + 1.0 - soil_t) * (1 - 0.4 * cov)
            soil_rows.append((pid, ts, *theta.round(3), soil_t, day))
        # --- 수분 스트레스 계수 Ks ---
        raw = float(np.sum(root_w * np.clip((theta - wp_v) / (fc_v - wp_v), 0, 1)))
        ks = float(np.clip(raw / 0.55, 0, 1))
        ks_hist.append(ks)
        # --- 바이오매스 (RUE) ---
        tm = df_wd.at[day, "tmean"]
        ft = float(np.clip(tm / 8, 0, 1))
        if sg[10] <= g < sg[92]:
            W += 2.3 * 0.48 * df_wd.at[day, "rs_mj"] * (1 - np.exp(-0.5 * lai)) * ks * ft * pm
        # --- 초장 ---
        s_now = smooth((g - sg[30]) / (sg[65] - sg[30])) if g > sg[30] else 0.0
        H += (s_now - s_prev) * (h_max - 15) * (0.6 + 0.4 * ks)
        if g < sg[30]:
            H = 5 + 10 * smooth((g - sg[10]) / (sg[30] - sg[10])) if g > sg[10] else 5.0
        s_prev = s_now
        # --- 엽 질소 (희석곡선 + 추비 + 노화) ---
        leaf_n = 4.4 * (max(W / 100, 1.5) / 1.5) ** -0.28  # N 희석곡선
        leaf_n += sum(0.35 * np.exp(-(day - f).days / 25) for f in fert_dates if day >= f)
        if g >= sg[65]:
            leaf_n *= 0.7 + 0.3 * cov / cover_max
        daily_state.append(dict(plot_id=pid, date=day, gdd=g, cover=cov, lai=lai, ks=ks, biomass=W,
                                height=H, leaf_n=float(np.clip(leaf_n, 1.5, 5.0))))

df_ds = pd.DataFrame(daily_state)

# ---------------- 3. 토양 센서 출력 (노이즈·결측·이상치·드리프트·통신두절) ----------------
df_s = pd.DataFrame(soil_rows, columns=["plot_id", "ts", "t10", "t30", "t60", "soil_t", "day"])
n = len(df_s)
for c, out in [("t10", "vwc_10cm"), ("t30", "vwc_30cm"), ("t60", "vwc_60cm")]:
    df_s[out] = df_s[c] + rng.normal(0, 0.4, n)
df_s["soil_temp_10cm"] = df_s["soil_t"] + rng.normal(0, 0.2, n)
fert_boost = sum(np.where(df_s["day"] >= f, 0.25 * np.exp(-(df_s["day"] - f).dt.days / 20), 0) for f in fert_dates)
df_s["soil_ec"] = 0.25 + 0.03 * df_s["t10"] + fert_boost + rng.normal(0, 0.03, n)

dq_events = []
drift = (df_s.plot_id == "P07") & FAULTS & (df_s.ts >= "2025-03-01") & (df_s.ts < "2025-03-15")
df_s.loc[drift, "vwc_10cm"] += np.linspace(0, 4, drift.sum())
dq_events.append(dict(plot_id="P07", start="2025-03-01", end="2025-03-14", variable="vwc_10cm", type="sensor_drift(+0~4%)"))
outage = (df_s.plot_id == "P15") & FAULTS & (df_s.ts >= "2025-04-22") & (df_s.ts < "2025-04-25")
dq_events.append(dict(plot_id="P15", start="2025-04-22", end="2025-04-24", variable="all", type="logger_offline"))
spike = (rng.random(n) < 0.002) & FAULTS
df_s.loc[spike, "vwc_10cm"] *= 1.8
keep = (rng.random(n) >= 0.03) & ~outage
df_s["is_spike"] = spike
soil_out = df_s.loc[keep, ["plot_id", "ts", "vwc_10cm", "vwc_30cm", "vwc_60cm", "soil_temp_10cm", "soil_ec"]].copy()
soil_out["timestamp"] = soil_out.pop("ts").dt.strftime("%Y-%m-%dT%H:%M")
soil_out = soil_out[["plot_id", "timestamp", "vwc_10cm", "vwc_30cm", "vwc_60cm", "soil_temp_10cm", "soil_ec"]].round(
    {"vwc_10cm": 1, "vwc_30cm": 1, "vwc_60cm": 1, "soil_temp_10cm": 1, "soil_ec": 2})
soil_out.to_csv(f"{OUT}/soil_sensor.csv", index=False)
dq_events.append(dict(plot_id="ALL", start="", end="", variable="vwc_10cm",
                      type=f"random_spike x1.8 ({int((spike & keep).sum())} rows)"))

# ==========================================================
# 7. 수확 (harvest.csv) - 날짜 먼저 산정 (이후 캐노피 측정 중단)
# ==========================================================
harv_rows, harvest_date = [], {}
truth_ks = []
for _, p in df_plots.iterrows():
    pid, cv = p.plot_id, p.cultivar
    d = df_ds[df_ds.plot_id == pid]
    sg = stage_gdd[pid]
    ks_til = d[(d.gdd >= sg[30]) & (d.gdd < sg[65])].ks.mean()  # 절간신장~개화: 이삭 생존기
    ks_gf = d[(d.gdd >= sg[65]) & (d.gdd < sg[92])].ks.mean()
    mat = pd.Timestamp(df_ph[(df_ph.plot_id == pid) & (df_ph.zadoks_stage == 92)].date.iloc[0])
    hd = min(mat + pd.Timedelta(days=int(rng.integers(7, 13))), W_END.normalize())
    harvest_date[pid] = hd
    w_fin = d[d.date <= mat].biomass.iloc[-1]
    pm = float(df_truth.loc[df_truth.plot_id == pid, "prod_mult"].iloc[0])
    hi = (0.47 if cv == "CV_A" else 0.45) - 0.14 * (1 - ks_gf) + rng.normal(0, 0.01)
    yld = w_fin * hi / 100 * (1 + rng.normal(0, 0.03))  # 수확 측정오차
    harv_rows.append(dict(plot_id=pid, harvest_date=hd.strftime("%Y-%m-%d"),
                          grain_yield=round(yld, 2),
                          tkw=round((46 if cv == "CV_A" else 43) - 13 * (1 - ks_gf) + rng.normal(0, 1.2), 1),
                          spikes_m2=int(round((200 + 330 * ks_til) * pm + rng.normal(0, 20))),
                          grain_protein=round(11.2 + 4.2 * (1 - ks_gf) + rng.normal(0, 0.35), 1),
                          harvest_index=round(hi, 3)))
    truth_ks.append(dict(plot_id=pid, ks_mean_stem_elong_to_anthesis=round(ks_til, 3),
                         ks_mean_grainfill=round(ks_gf, 3), final_biomass_true=round(w_fin, 1),
                         stress_days_ks_lt_0_5=int((d.ks < 0.5).sum())))
df_h = pd.DataFrame(harv_rows)
df_h.to_csv(f"{OUT}/harvest.csv", index=False)

# ==========================================================
# 4. 캐노피 (canopy.csv) - 주 1회, 13시 측정(일중 VPD 최대 부근)
# ==========================================================
cn = []
noon = df_w.set_index("timestamp")
for c_date in pd.date_range("2024-12-01", "2025-07-10", freq="7D"):
    vpd12, t12 = noon.at[c_date + pd.Timedelta(hours=13), "vpd"], noon.at[c_date + pd.Timedelta(hours=13), "t_air"]
    for _, p in df_plots.iterrows():
        if c_date > harvest_date[p.plot_id]:
            continue
        s = df_ds[(df_ds.plot_id == p.plot_id) & (df_ds.date == c_date)].iloc[0]
        cov, ks = s.cover, s.ks
        ndvi = 0.14 + 0.74 * cov ** 0.8 + rng.normal(0, 0.015)
        spad = 22 + 7 * s.leaf_n + rng.normal(0, 1.2)
        ndre = 0.08 + 0.45 * cov * (0.6 + 0.4 * (s.leaf_n - 1.5) / 3.5) + rng.normal(0, 0.012)
        ctd = cov * (ks * (-0.5 - 1.4 * vpd12) + (1 - ks) * (1.0 + 1.3 * vpd12)) + (1 - cov) * 1.0 + rng.normal(0, 0.4)
        if rng.random() < 0.03:
            continue  # 결측 3%
        cn.append(dict(plot_id=p.plot_id, date=c_date.strftime("%Y-%m-%d"),
                       ndvi=round(float(np.clip(ndvi, 0.1, 0.92)), 3), ndre=round(float(np.clip(ndre, 0.05, 0.62)), 3),
                       canopy_temp=round(t12 + ctd, 1), ctd=round(ctd, 2),
                       lai=round(max(0.0, s.lai + rng.normal(0, 0.12)), 2),
                       plant_height=round(max(3.0, s.height + rng.normal(0, 1.5)), 1),
                       spad=round(float(np.clip(spad, 28, 55)), 1), source="synthetic"))
df_c = pd.DataFrame(cn)
out_idx = rng.choice(len(df_c), 4, replace=False)  # 캐노피 이상치 4건 (드론 반사보정 오류 가정)
df_c.loc[out_idx, "ndvi"] = (df_c.loc[out_idx, "ndvi"] * 0.5).round(3)
dq_events.append(dict(plot_id="various", start="", end="", variable="ndvi",
                      type="reflectance_error x0.5 at rows " + ",".join(map(str, sorted(out_idx)))))
df_c.to_csv(f"{OUT}/canopy.csv", index=False)

# ==========================================================
# 6. 파괴 샘플링 (biomass.csv) - Z30, Z65, Z92
# ==========================================================
bm = []
for r in df_ph[df_ph.zadoks_stage.isin([30, 65, 92])].itertuples():
    s = df_ds[(df_ds.plot_id == r.plot_id) & (df_ds.date == pd.Timestamp(r.date))].iloc[0]
    bm.append(dict(plot_id=r.plot_id, date=r.date, zadoks_stage=r.zadoks_stage,
                   biomass_dry=round(max(50.0, s.biomass * (1 + rng.normal(0, 0.07))), 1),
                   leaf_n=round(float(np.clip(s.leaf_n + rng.normal(0, 0.12), 1.5, 5.0)), 2)))
pd.DataFrame(bm).to_csv(f"{OUT}/biomass.csv", index=False)

# ==========================================================
# 정답값 저장 (검증 전용)
# ==========================================================
df_truth.merge(pd.DataFrame(truth_ks), on="plot_id").to_csv(f"{TRUTH}/plot_truth.csv", index=False)
pd.DataFrame(dq_events).to_csv(f"{TRUTH}/data_quality_events.csv", index=False)
df_ds.assign(date=df_ds.date.dt.strftime("%Y-%m-%d")).round(4).to_csv(f"{TRUTH}/daily_crop_state_true.csv", index=False)
print("생성 완료:", OUT)
