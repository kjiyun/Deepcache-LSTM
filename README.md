# TTLCacheNet: LSTM 기반 캐시 TTL 최적화 시스템

## 📌 개요
**TTLCacheNet**는 요청 패턴을 학습해 미래의 객체 요청을 예측하고,
이를 바탕으로 캐시 유지 시간(TTL, Time-To-Live)을 자동으로 추천하는 시스템입니다.

기존의 LRU 캐시처럼 과거만 보는 방식에서 벗어나,
LSTM 기반 딥러닝 모델을 활용해 예측 기반 캐싱 정책을 실현합니다.

## 🎯 목표
- Redis 기반 시스템에 쉽게 연동 가능한 TTL 추천 API 개발
- 캐시 메모리 활용률 향상
- Hit Ratio 상승 및 응답 속도 개선

## 🧠 핵심 구성요소
### 1. 요청 데이터 생성
- generateSyntheticDataset.py: Zipf 분포 기반 synthetic workload 생성 (Dataset1)
- generateMediSynDataset.py: 실시간 요청 변화 반영한 workload 생성 (Dataset2)

### 2. 전처리
- requestAnalysis.py
    → 요청 로그를 기반으로 시간대별 bin, 객체 속성(frequency, lifespan 등)을 추출

### 3. LSTM Encoder-Decoder 모델 학습
- 입력: 과거 20시간
- 출력: 미래 10시간 또는 26시간

### 4. TTL 추천
```
ttl = (predicted_prob / predicted_prob.max()) * max_ttl
```
- 요청 확률이 높을수록 TTL을 길게 설정
- Redis에 적용 가능한 TTL 값 반환

### 5. 캐시 시뮬레이션
- LRU vs DeepCache TTL 기반 캐시 비교
- 성능 지표: Cache Hit Ratio

## 실험 결과
<img src=https://github.com/user-attachments/assets/e7e3062f-c33a-4812-9548-21e2f3f93ee7 width=40% height=40%>

## 👩‍💻 Contributors
- 기반 논문: DeepCache: A Deep Learning Based Inference Caching Framework for Content Delivery
