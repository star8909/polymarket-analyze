# CLAUDE.md

이 파일은 Claude Code (claude.ai/code) 가 이 저장소에서 작업할 때 참고하는 가이드.

## 프로젝트 목적

**Polymarket 예측시장**을 **확률분포로 다루는** 분석/베팅 플레이그라운드.
시장 가격 = 시장이 매기는 확률. 시장 가격 ≠ 진짜 확률인 마켓을 체계적으로 찾아 +EV 베팅.

자매 프로젝트 (모두 같은 방법론 — CVaR + GMM + walk-forward + 분포 기반):
- `c:\Projects\ProbabilityDistribution` — 암호화폐 (현물 + funding 차익)
- `c:\Projects\EquityDistribution` — 한투 해외주식 (R70 메인 챔, BT 11.74)
- `c:\Projects\FuturesDistribution` — 한투 해외선물옵션 (Agri momentum baseline)
- `c:\Projects\polymarket-analyze` — Polymarket 예측시장 (이 프로젝트)

## 핵심 가설

**도메인 지식 없이 통계만으로 +EV.** 정치/스포츠/크립토 깊이 몰라도 패턴만으로 이김.

작동하는 이유 — 시장에 비합리성이 구조적으로 박혀 있음:

1. **롱샷 편향 (Longshot bias)** — 5센트 마켓에 로또처럼 베팅 → 진짜 확률보다 가격 높음. 통계적 사실, 매번 반복.
2. **감정 오버슈팅** — 뉴스 발생 시 가격 과잉반응 후 회귀. 어떤 뉴스인지 몰라도 z-score만 보면 됨.
3. **수학적 비일관성 (Sum-to-1 위반)** — multi-outcome 마켓에서 모든 Yes 가격 합 ≠ $1. 사람들이 마켓 여러 개 동시에 안 봐서 발생. 순수 산수.
4. **카테고리별 편향 차이** — 정치 vs 스포츠 vs 크립토 calibration 다름.

## 데이터 출처

### 1차 — 빠른 시작 (warproxxx/poly_data 스냅샷)
```bash
git clone https://github.com/warproxxx/poly_data
# README 따라 snapshot 다운로드
# CSV/parquet으로 종료 마켓 + 거래 내역 즉시 활용 (2일 절약)
```

### 2차 — 직접 수집 (자체 자산화)
- **Gamma API** (`gamma-api.polymarket.com`) — 마켓 메타·이벤트·태그·검색. **인증 불필요, 무료**
- **CLOB API** (`clob.polymarket.com`) — 오더북·가격 히스토리·중간가·스프레드
- **Data API** — 유저 포지션·거래·홀더·OI·리더보드

### 3차 — 온체인 (Polygon)
- **Bitquery** (GraphQL) — `PredictionTrades`, `PredictionSettlements`. **realtime 7일치 한정**
- **Goldsky 서브그래프** — 무제한 백필 (`orderFilled` 이벤트). 대량 백테스팅 적합.

## 폴더 구조

```
polymarket-analyze/
├── src/                       # 코어 (data_loader, config, polymarket_universe)
├── iters/                     # iter*.py 분석 스크립트
├── tools/                     # fetch_closed.py, fetch_active.py, snapshot 처리
├── results/                   # iter*.json (commit, 재현성)
├── logs/                      # iter*.log
├── dashboard/                 # 챔피언 대시보드
├── archive/                   # 옛 파일
├── data/                      # 캐시 (gitignored)
│   ├── cache/                 # parquet 마켓 데이터
│   └── snapshot/              # warproxxx/poly_data 스냅샷
├── CLAUDE.md / README.md / RESULTS.md / requirements.txt
```

**중요**: `iters/` 와 `tools/` 의 .py 파일들은 `from src import ...` 사용 → **항상 root 에서 실행**.

## 분석 전략 (iter 진행 순서)

### Phase 1 — Calibration (가장 강력한 무기)
종료된 마켓 결과로 **price → actual outcome rate** 곡선 측정.

```
가격 0.05~0.10 마켓 1000개 → 실제 Yes 종료율 = 3% (시장 과대평가)
가격 0.45~0.55 → 50% (시장 정확)
가격 0.85~0.95 → 91% (시장 약간 과소평가)
```

이 한 번 측정으로 **자동 베팅 가능**. 5센트 마켓 무조건 No 매수 → 통계적으로 이김.

### Phase 2 — Sum-to-1 Arbitrage (순수 산수)
multi-outcome 마켓 (예: "2026 대선 후보 X명") 모든 Yes 가격 합산:
- 합 < 1 → 전부 Yes 매수 (수익 보장)
- 합 > 1 → 전부 No 매수 (수익 보장)

매일 수십 개씩 발생. 자동화로 스캔.

### Phase 3 — Time-series 분포 (변동성 mean reversion)
각 마켓 가격 p(t) 시계열을:
- **Beta 분포** — 0~1 범위 자연 prior
- **Logit 변환 후 정규성** — `log(p/(1-p))` z-score
- **GARCH** — 변동성 클러스터링 (이벤트 직전 폭발)

가격이 ±2σ 극단 진입 → mean reversion 베팅 후보.

### Phase 4 — 조건부 일관성 (cross-market arbitrage)
- "트럼프 대통령 당선" Yes vs "트럼프 공화당 후보" Yes → 전자 > 후자면 모순
- "BTC 10만 12/31" vs "BTC 12만 12/31" → 후자 가격 ↑ 면 차익

### Phase 5 — Kelly Sizing
```
f* = (p × b - (1-p)) / b
```
- p = 본인 추정 진짜 확률 (calibration 기반)
- b = 배당 (= (1-price)/price)
- Fractional Kelly (0.25~0.5×) 적용해서 추정 오차 흡수

## 주요 명령

Windows 콘솔에서 한글 출력. **항상** `PYTHONIOENCODING=utf-8`.

```bash
# 1) 종료 마켓 데이터 수집 (calibration용 ground truth)
PYTHONIOENCODING=utf-8 python tools/fetch_closed.py

# 2) 활성 마켓 스냅샷 (sum-to-1 스캐너용)
PYTHONIOENCODING=utf-8 python tools/fetch_active.py

# 3) 가격 시계열 (분포 분석용)
PYTHONIOENCODING=utf-8 python tools/fetch_prices.py --token-id <id>

# 4) iter*.py 분석
PYTHONIOENCODING=utf-8 python iters/iter01_fetch_closed.py
PYTHONIOENCODING=utf-8 python iters/iter02_calibration.py
PYTHONIOENCODING=utf-8 python iters/iter03_sum_to_one_scanner.py
```

## 아키텍처

### 데이터 흐름

```
warproxxx snapshot ─┐
Gamma API ──────────┤
CLOB API ───────────┤
Goldsky 서브그래프 ─┘
                    ▼
           src/data_loader.py (parquet 캐시)
                    ▼
           src/calibration.py / src/distributions.py
                    ▼
           Strategy.scan() / Strategy.bet_decision()
                    ▼
           backtester (수수료 0bps, 가스 ~0.5%, 슬리피지 ~1%)
                    ▼
           CLOB API (실거래 시) / paper trading log
```

### 비용 모델 (선물/주식과 다름)

| 항목 | 값 |
|-----|-----|
| Polymarket 수수료 | **0%** (현재) |
| Polygon 가스비 | ~$0.01 (negligible) |
| 스프레드 | 1~5% (낮은 유동성 마켓에서 큼) |
| 슬리피지 | 1~3% (depth 작은 마켓) |
| **합산 baseline** | **2~5%** per trade |

**규칙**: 백테스트는 `fee_bps=200` 보수적, stress test `300/500bps`까지.

### 봇/전략 계층 (계획)

```
src/strategies.py
    BaseStrategy
    ├── CalibrationStrategy        # 5센트 No, 95센트 Yes 자동 베팅
    ├── SumToOneStrategy           # multi-outcome arbitrage
    ├── MeanReversionStrategy      # ±2σ 극단 → reversion 베팅
    ├── CrossMarketStrategy        # 조건부 일관성 위반
    └── KellySizing                # Fractional Kelly (0.25x)
```

## Critical conventions

- **항상 `PYTHONIOENCODING=utf-8`** — Windows 한글 print
- **결과 누출 마켓 필터** — 해결까지 1주일 미만 마켓은 false signal 많음. 최소 7일 남음 + 거래량 필터
- **소량 마켓 fitting 금지** — 거래 5건짜리 마켓에 분포 fit하면 노이즈만. 최소 거래량 1k 달러 + 홀더 50명+
- **UMA 오라클 분쟁 risk** — 마켓 무효화 가능성 항상 존재. 꼬리 위험으로 인지
- **카테고리별 편향 분리** — 정치/스포츠/크립토 calibration 따로 학습
- **Walk-forward만 신뢰** — 자매 프로젝트 (iter14 ML invert 가짜 챔피언) 교훈

## 실거래 안전 규칙

1. **Paper trading 4주 이상** — 백테스트 vs 실거래 갭 확인
2. **자본 5% 이하만** — Polymarket은 외환 손실 + 가스 + UMA risk
3. **Fractional Kelly 0.25x** — 추정 오차 흡수
4. **마켓당 max $100** — 단일 마켓 위험 캡
5. **카테고리 분산** — 정치/스포츠/크립토 한 곳 몰빵 X
6. **자동화 필수** — 24h 마켓에서 수동 모니터링 X

## Iteration history

새 iter 결과는 `RESULTS.md` 또는 `dashboard/index.html` 에 한 줄 추가.
iter*.json + iter*.log 은 `results/` / `logs/` 에 commit.

iter 번호 진행:
- iter01: closed markets 수집 (Gamma API + warproxxx 스냅샷 통합)
- iter02: calibration 곡선 분석 (가격 vs 실제 결과율)
- iter03: sum-to-1 arbitrage 스캐너
- iter04+: time-series 분포, cross-market 조건부, Kelly sizing 최적화
