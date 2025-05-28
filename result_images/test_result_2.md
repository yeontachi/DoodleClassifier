## 테스트 결과 요약 보고서
 - 총 테스트 이미지 : 30장
 - 이미지 경로: test_images/
 - 이미지 파일명: 각 클래스 이름과 동일(예: airplane.png, car.png 등)

### Basic CNN model(basic_cnn_model_96.h5)
 - 일치 : 1개
 - 불일치 : 29개
 - 정확도 : 3.33%
 - 예측 결과 이미지 > test2_basicCNN_model.png

### MobileNetV2 model(mobilenet_model_96.h5)
 - 일치 : 11개
 - 불일치 : 18개
 - 정확도 : 36.67%
 - 예측 결과 이미지 > test2_mobilenet_model(96x96).png

## 이전 테스트(test1) 결과와 비교 
- Basic CNN: 6.67% -> 3.33%
- MobileNetV2: 26.67% -> 36.67%
- 이전 모델과 차이점

| 항목                          | 기존 모델 (28x28)                                 | 개선된 모델 (96x96)                                   |
| --------------------------- | --------------------------------------------- | ------------------------------------------------ |
| 🔸 입력 이미지 크기                | `28x28` (1채널, 흑백)                             | `96x96` (1채널, 흑백)                                |
| 🔸 CNN 구조                   | 기본 3층 CNN<br>작은 필터와 풀링                        | CNN 필터 수 증가 및 깊이 확장                              |
| 🔸 MobileNetV2 입력 처리        | 28x28 → 3채널 변환 → `UpSampling(8x)` → `224x224` | 96x96 → 3채널 변환 → `UpSampling(2x)` → `192x192`    |
| 🔸 MobileNetV2 Base Model   | `input_shape=(224, 224, 3)`                   | `input_shape=(192, 192, 3)`                      |
| 🔸 데이터 전처리                  | 28x28 픽셀로 reshape                             | 96x96 픽셀로 resize 후 reshape                       |
| 🔸 학습 데이터 수                 | 클래스당 2000개                                    | 클래스당 2000개 (동일)                                  |
| 🔸 데이터 증강                   | 회전, 이동, 확대                                    | 동일 (ImageDataGenerator)                          |
| 🔸 EarlyStopping & ReduceLR | 적용                                            | 동일                                               |
| 🔸 기본 CNN 성능                | 정확도 약 **3.3%**                                | 개선 없음 (성능 유지)                                    |
| 🔸 MobileNetV2 성능           | 정확도 약 **36.7%**                               | 확연한 성능 개선                                        |
| 🔸 클래스 편향 현상                | 있음 (book, laptop 쏠림)                          | MobileNetV2에서는 완화됨                               |
| 🔸 모델 저장명                   | `basic_cnn_model.h5`, `mobilenet_model.h5`    | `basic_cnn_model_96.h5`, `mobilenet_model_96.h5` |


## 결과
확률값이 높은 예측도 있지만, 일부 클래스에서는 혼동이 많다.(예: Whale -> fish)
-> 전이학습 기반 MobileNetV2가 기본 CNN보다는 월등히 우수함

| 항목         | Basic CNN  | MobileNetV2 |
| ---------- | ---------- | ----------- |
| 정확도        | 3.33%      | 36.67%      |
| 추론 속도      | 빠름 (경량)    | 느림 (복잡한 구조) |
| 예측 클래스 다양성 | 매우 부족      | 비교적 균형적     |
| 실전 사용 가능성  | ❌ (재학습 필요) | 🔄 (튜닝 필요)  |


