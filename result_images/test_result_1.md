## [모델 및 테스트 환경 정보]

- 테스트 이미지 수: 30장
- 클래스 수: 30개
   - whale, car, tree, cat, airplane, hat, dog, fish, bicycle, house,
     flower, star, moon, clock, cloud, candle, cup, book, bus, camera,
     chair, door, guitar, hamburger, ice cream, key, laptop, pencil, shoe, cake

- 이미지 전처리
   - 흑백(GRAYSCALE) 변환
   - 크기 조정: 28x28
   - 정규화: 0~1 범위로 나눔 (float32)

- 입력 텐서 형태
   - Basic CNN: (28, 28, 1)
   - MobileNetV2: (28, 28, 1) → (224, 224, 3)로 업샘플링 + 채널 3배

- 출력 형태
   - 소프트맥스(softmax) → 30 클래스 중 하나로 확률 분포 반환

- 평가 지표
   - Top-1 정확도 (예측과 파일명 비교)
   - 확률 출력 (예측된 클래스의 softmax 값)

- 시각화 방식
   - 각 이미지당 예측 결과 표시 (정답일 경우 초록색, 오답일 경우 빨간색)
   - 예측 클래스명, 확률, 일치 여부 출력

- 모델 저장 위치
   - basic_cnn_model.h5
   - mobilenet_model.h5

- 테스트 이미지 경로
   - ./test_images/


## [ 정확도 ]
- Basic CNN: 6.67%
- MobileNetV2: 26.67%
