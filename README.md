# WheatDT — Wheat Field-Trial Digital Twin

Reference implementation of a trial-aware digital twin for wheat field trials (Hoseo University, SmartFarm AI Data Laboratory; Korea–EU Horizon Europe precision-agriculture project). Version 1.1.2.

All data in `data/` and all model training data are **synthetic**, produced by the platform's own crop–soil generator. Accuracy figures verify the pipeline; they are not agronomic validation.

```bash
pip install -r requirements.txt
python -m wheatdt run --data data --out dashboard.html   # open dashboard.html in a browser
```

| Folder | Contents |
|---|---|
| `wheatdt/` | Platform package (ingest, metadata, QC, indicators, forecasting, twin, dashboard) |
| `models/` | Trained forecast models + `model_report.json` |
| `data/` | Demo synthetic season (seed 42); `data/_truth/` holds the generator's ground truth |
| `training_seasons/` | 34 synthetic training/test seasons (zipped) + the feature tables the models were trained on |
| `benchmark/` | Scripts and result tables for the paper's QC benchmark |

---

# Wheat Trial Digital Twin (`wheatdt`)

ECICE 2026 논문 *Experimental-Design Metadata Enables Sensor Drift Detection in a Wheat Field-Trial Digital Twin* 에서 제안한 플랫폼의 구현체입니다.
Reference implementation of the platform described in the ECICE 2026 paper (Lim, Baslam & Hwang).

## 구조 (논문 Figure 1과 동일)

| 계층 | 모듈 | 하는 일 |
|---|---|---|
| Data sources / generator | `wheatdt/synthetic.py` | 가상 시즌 생성(8개 CSV + `_truth/` 정답) |
| Ingestion | `wheatdt/ingest.py` | 8개 테이블 스키마 검증, 범위 점검 |
| Design metadata | `wheatdt/metadata.py` | 블록·처리구·품종·토성·FC/WP 보관, 피어 그룹 생성 |
| Trial-aware QC | `wheatdt/qc.py` | 스파이크, 결측, 드리프트(SaQC DFN, 처리구 그룹, 유효수분), NDVI 이상 |
| Crop indicators | `wheatdt/indicators.py` | 근권 유효수분(RAW), 적산온도, Zadoks 단계, 경보 |
| Forecasting (v1.1) | `wheatdt/features.py`, `models.py`, `train.py` | 7일 근권 수분 전망·조기 경보, 생육 단계별 수량 예측 |
| Virtual twin | `wheatdt/twin.py` | 위 결과를 포장별 상태로 묶고 날짜별 스냅샷 제공 |
| Dashboard | `wheatdt/dashboard.py` | 서버 없이 열리는 단일 HTML 대시보드 생성 |

## 실행

```bash
pip install pandas numpy saqc          # saqc가 없으면 peer-trend 규칙으로 자동 대체
python -m wheatdt generate --out data --seed 42
python -m wheatdt run --data data --out dashboard.html
python -m wheatdt snapshot 2025-05-20 --data data
python -m wheatdt train            # 예측 모델 재학습 (training_seasons/ 의 34개 시즌 사용)
```

`models/`에 학습된 모델이 들어 있어 바로 예측이 켜집니다. 이 폴더가 없으면 예측 없이 QC·지표만 동작합니다.

Python에서:
```python
from wheatdt import DigitalTwin
twin = DigitalTwin("data").update()
twin.snapshot("2025-05-20")      # 포장별 상태 + 경보
twin.evaluate()                  # 가상 정답과 비교 (실측 데이터에는 _truth/가 없으므로 None)
```

## 실측 데이터로 전환

1. 컨소시엄 데이터를 같은 8개 CSV 형식(`ingest.SCHEMA`)으로 `data/`에 넣습니다.
2. `wheatdt/config.py`의 `SOIL_HYDRAULICS`를 실측 포장용수량(FC)·위조점(WP)으로 바꿉니다. 드리프트 탐지가 이 값에 의존합니다.
3. `canopy.csv`의 `source`를 `measured`로 두면 대시보드의 Synthetic 표시가 사라집니다.
4. 실측 데이터가 쌓이면 `config.QC` 임계값을 다시 보정합니다.

## 기본 설정과 근거

- 드리프트 기본 검사: SaQC `flagDriftFromNorm`, 7일 창, spread 3.0, 처리구 그룹(8개 포장), 유효수분 = 100 × (θ − WP)/(FC − WP). 논문 Table 4 마지막 줄(F1 0.899).
- 보조 검사: peer-trend 규칙(토양 × 처리구 그룹), 그룹 절반 이상이 함께 움직이는 날은 현장 변화로 보고 제외.
- 스파이크: FC + 10% 초과 또는 13시간 이동중앙값 대비 4%p 이상 차이(5개 시즌에서 튜닝).
- 플래그는 값 옆에 저장하며 값은 삭제하지 않습니다.

## 범위와 한계

센서 → 가상 모델은 자동, 현장 조치는 경보를 보고 사람이 실행합니다(엄밀한 정의로는 human-in-the-loop 디지털 섀도). 자동 관개 제어와 AI 예측 모델은 아직 포함하지 않았습니다.

## 예측 모델 (v1.1)

| 예측 | 입력 (예측 시점까지의 정보만 사용) | 방법 |
|---|---|---|
| 7일 후 근권 유효수분 | 현재·최근 RAW, 생육 단계, NDVI(피복), 최근 7일 ET₀·강우·기온, 향후 7일 관개 계획, ET₀ 기후값 | 물수지 투영 + gradient boosting (하이브리드) |
| 수분 스트레스 조기 경보 | 위 예측값 | 학습 시즌에서 고른 규칙: **hybrid** (임계값 0.505) |
| 수량 (Z30, Z65, Z65+21일) | 누적 수분 스트레스, NDVI·LAI·CTD, 관개량, 처리구·품종·토성·블록 | 학습 시즌 교차검증으로 평균/ridge/boosting 중 선택 |

학습 24개 시즌(seed 201–224), 평가 10개 시즌(seed 301–310, 학습에 미사용). 강우 예보는 쓰지 않고 ET₀ 기후값만 씁니다.

| 7일 수분 전망 | 평균 오차 (%p) | 조기 경보 F1 |
|---|---|---|
| Persistence (no change) | 7.1 | 0.0 |
| Water-balance projection | 7.0 | 0.772 |
| Hybrid model (water balance + gradient boosting) | 4.6 | 0.757 |
| Early warning rule used by the platform | – | 0.757 |

| 수량 예측 시점 | 처리구 평균 RMSE | 플랫폼 모델 RMSE (R²) |
|---|---|---|
| Z30 | 0.607 | 0.673 (0.77), Linear (ridge) |
| Z65 | 0.607 | 0.461 (0.892), Linear (ridge) |
| Z65+21 | 0.607 | 0.455 (0.895), Linear (ridge) |

해석:
- 7일 수분 전망은 하이브리드 모델이 가장 정확합니다. 조기 경보 성능은 단순 물수지 투영과 비슷합니다.
- 수량 예측은 개화기(Z65) 이후에야 처리구 평균보다 나아집니다. 절간신장기(Z30) 예측은 처리구 평균보다 못하므로 참고용으로만 쓰십시오.
- 가상 데이터는 플랫폼 자체 생성기에서 나왔으므로 위 성능은 파이프라인 검증이자 상한값입니다. 농학적 검증이 아닙니다. 실측 데이터로 재학습·재평가가 필요합니다.
