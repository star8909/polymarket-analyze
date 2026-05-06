# Polymarket 자동거래 셋업 가이드

calibration 기반 자동 베팅 봇. **paper-first** → 검증 후 소액 라이브.

---

## 0. 사전 준비물

| 항목 | 비용 | 비고 |
|------|------|------|
| Python 3.10+ | 무료 | 이미 있음 |
| Polygon 지갑 | 무료 | MetaMask 또는 Polymarket Magic Wallet |
| MATIC (가스비) | ~$1 | 가스비용 |
| USDC.e (Polygon) | $50~$1000 | 시작 자본 |

---

## 1. 지갑 생성 (3가지 옵션)

### A. Polymarket 웹 가입 (가장 쉬움 - 추천)

1. polymarket.com 접속
2. **Email로 가입** → Magic Wallet 자동 생성 (Polygon 주소)
3. Profile → Settings → Export Private Key
4. **`signature_type=1`** 사용 (이메일 지갑)

### B. 기존 MetaMask 사용

1. MetaMask에 Polygon 네트워크 추가
   - Network Name: `Polygon`
   - RPC URL: `https://polygon-rpc.com`
   - Chain ID: `137`
   - Symbol: `MATIC`
   - Explorer: `https://polygonscan.com`
2. **`signature_type=0`** 사용
3. polymarket.com 접속 → Connect MetaMask
4. **USDC + CTF 컨트랙트 approval** 1회 필요 (자동 prompt)

### C. 하드웨어 지갑 (Ledger/Trezor)

- B와 동일 + MetaMask에 Ledger 연결

---

## 2. USDC.e 입금 (Polygon)

**중요**: Polymarket은 **USDC.e (bridged)** 만 받음. **USDC (native)** 와 다름!

### 한국 거래소 → Polygon

| 경로 | 절차 |
|------|------|
| **업비트** | USDT 매수 → 폴리곤 출금 → Sushi/Quickswap 에서 USDC.e로 swap |
| **바이낸스** | USDC 매수 → Polygon 네트워크 출금 → bridge 안 거치면 native USDC, 거치면 USDC.e (확인 필수) |
| **Coinbase** | USDC 매수 → Polygon 출금 → 자동 USDC.e |

### 검증

지갑 주소 → Polygonscan → Tokens 탭에서:
- Symbol: `USDC.e` (예전 이름 USDC, contract `0x2791bca1f2de4661ed88a30c99a7a9449aa84174`)
- 잔액 > $0

---

## 3. py-clob-client 설치

```bash
cd c:/Projects/Quant/polymarket-analyze
pip install py-clob-client python-dotenv
```

---

## 4. .env 작성

`live/.env` 파일 (gitignore 필수!):

```bash
# Polymarket 인증
PRIVATE_KEY=0x_여기에_private_key_64자
FUNDER_ADDRESS=0x_여기에_proxy_or_eoa_address
SIGNATURE_TYPE=1  # 1=email magic wallet, 0=metamask EOA

# CLOB endpoint
CLOB_HOST=https://clob.polymarket.com
CHAIN_ID=137

# 안전장치
MAX_BET_USD=20             # 마켓당 최대
DAILY_LOSS_LIMIT_USD=100   # 일일 손실 한도
EDGE_THRESHOLD_PP=5        # 5pp 이상만 베팅
KELLY_FRACTION=0.25        # Fractional Kelly
KILL_SWITCH_FILE=KILL      # 이 파일 생기면 즉시 정지
```

---

## 5. 첫 동작 확인 (read-only)

```bash
python live/check_connection.py
```

→ 잔고 + 첫 마켓 가격 출력. 에러 없으면 OK.

---

## 6. paper trading 시작

```bash
python live/paper_trade.py --capital 1000 --duration 28d
```

→ 28일간 가상 자본 $1000으로 시뮬레이션. 매일 PnL 기록.

---

## 7. 라이브 시작 (paper 4주 통과 후만)

```bash
python live/bot.py --capital 100 --kill-switch live/KILL
```

→ 실제 $100 USDC.e로 시작. **drawdown -10% 자동 정지**.

---

## 단계별 자본

| 단계 | 자본 | 기간 | 통과 기준 |
|------|------|------|---------|
| Paper | $1000 가상 | 4주 | Sharpe ≥ 0.5 |
| Live A | $100 실전 | 4주 | profit ≥ 0 |
| Live B | $500 실전 | 4주 | Sharpe ≥ 0.5 |
| Live C | $2000 실전 | 12주 | Sharpe ≥ 0.7 |
| 본격 | $5000+ | — | 단계별 분기 재검증 |

---

## 위험 인지

| 위험 | 어떻게 대응 |
|------|----------|
| **UMA 분쟁** | 카테고리 분산, 마켓당 $20 캡 |
| **유동성 부족** | volume 24h ≥ $5k 마켓만 |
| **결과 누설** | 종료 7일 미만 마켓 회피 |
| **API 다운** | 재시도 + 24h 모니터링 |
| **봇 버그** | 잔고 대조 5분마다, mismatch 1% 시 자동 정지 |
| **본인 실수** | KILL 파일 생성 = 즉시 정지 (Slack/cron 가능) |

---

## 다음 단계

이 디렉토리에 함께 생성된 파일:
- [polymarket_client.py](polymarket_client.py) — CLOB API 래퍼
- [decision_engine.py](decision_engine.py) — calibration 기반 의사결정
- [bot.py](bot.py) — 메인 실행 루프
- [paper_trade.py](paper_trade.py) — 페이퍼 시뮬레이터
- [kill_switch.py](kill_switch.py) — 비상 정지
