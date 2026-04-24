# Bubble

BTC/현금 배분을 위한 HMM 레짐 인지형 리스크 모델 엔진입니다. Bubble은 레짐 확률과 CF-VaR, 스트레스 사이징, CPPI, TIPP floor 제어를 결합해 시장 상태 변화에 따른 익스포저를 결정합니다.

백테스트는 리스크 엔진을 검증하기 위한 하네스이며, 프로젝트의 핵심은 리스크 모델입니다. 이 저장소는 리서치와 모델 평가 목적이며 투자 조언, 실거래 봇, 프로덕션 포트폴리오 운용 시스템이 아닙니다.

## 아키텍처

```
리스크 엔진
├── CF-VaR 99% (90일 롤링 + 2022 스트레스)
├── CPPI: 최대 포지션 = 쿠션 / CF-VaR
├── TIPP: Floor가 포트폴리오 고점의 60%까지 상향
├── Regime Reset: Bull 80%+ 5일 연속 + SMA50 상향 돌파
└── 수동 스트레스 오버라이드 (config)

시그널 레이어
└── HMM 레짐 감지
    5개 피처 → 3-상태 가우시안 HMM → 상승/횡보/하락
    피처:
    - BTC 추세
    - BTC 변동성
    - BTC MA200 괴리율
    - 나스닥 추세
    - 달러인덱스 추세

검증 하네스
└── 리스크 모델 평가용 현물 BTC / 현금 비중 배분 시뮬레이션
```

## 데이터 소스

| 데이터 | 출처 |
|--------|------|
| BTC OHLCV | Binance REST API |
| 나스닥 / 달러인덱스 | yfinance |

## 프로젝트 구조

```
├── config.py               # 공용 실행 설정
├── risk.py                 # CF-VaR, 스트레스, CPPI, TIPP 리스크 엔진
├── engine.py               # HMM 피처, 학습, 레짐 확률 엔진
├── backtest.py             # TIPP/Regime Reset 포함 검증 하네스
├── collect_data.py         # BTC + 매크로 데이터 수집
├── tests/                  # 엔진/리스크 스모크 테스트
├── LICENSE
└── requirements.txt
```

## 빠른 시작

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 히스토리 데이터 갱신 & 검증 하네스 실행
python3 collect_data.py
python3 backtest.py
```

## Docker

```bash
docker build -t bubble-risk-engine .

# 데이터를 받고 검증 결과 CSV를 로컬 data/에 저장
docker run --rm -v "$PWD/data:/app/data" bubble-risk-engine python collect_data.py
docker run --rm -v "$PWD/data:/app/data" bubble-risk-engine

# 테스트 실행
docker run --rm bubble-risk-engine python -m unittest
```

## 검증

```bash
python3 -m unittest
```

전체 검증 실행은 Binance와 yfinance에서 데이터를 받으므로 네트워크 접근이 필요합니다.

## 설정

실행에 쓰는 주요 값은 `config.py`에 모여 있습니다.

- `DATA_CONFIG`: 데이터 폴더, 수집 시작일, BTC 심볼, 매크로 티커
- `CORE_BACKTEST_CONFIG`: 메인 백테스트 자본, floor, 재학습 주기, 출력 경로
- `CORE_BACKTEST_CONFIG.manual_stress`: 수동 최악 일간 낙폭 오버라이드

`collect_data.py`는 이제 기존 CSV가 있어도 최신 구간까지 다시 받아오므로, 기본 백테스트 경로에서 예전 파일을 조용히 재사용하는 문제가 줄었습니다.

## 면책

이 프로젝트는 교육 및 리서치 목적입니다. 백테스트 결과는 미래 성과를 보장하지 않습니다. 이 코드는 주문 실행, 라이브 포지션 관리, 프로덕션 리스크 통제를 포함하지 않습니다.

## 라이선스

MIT License.

## 작성자

[@starboard03](https://github.com/starboard03)
