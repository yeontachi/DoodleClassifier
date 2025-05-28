## 테스트 결과 요약 보고서
 - 총 테스트 이미지 : 30장
 - 이미지 경로: test_images(94x94)/
 - 이미지 파일명: 각 클래스 이름과 동일(예: airplane.png, car.png 등)
 - 테스트 이미지 30장 실제 크기 94x94로 줄임

 ### Basic CNN model(basic_cnn_model_96.h5)
 - 일치 : 1개
 - 불일치 : 29개
 - 정확도 : 3.33%
 - 예측 결과 이미지 > test3_basicCNN_model.png

### MobileNetV2 model(mobilenet_model_96.h5)
 - 일치 : 9개
 - 불일치 : 21개
 - 정확도 : 30.00%
 - 예측 결과 이미지 > test3_mobilenet_model.png

## 이전 테스트(test1) 결과와 비교 
- Basic CNN: 3.33% -> 3.33% 동일일
- MobileNetV2: 36.67% -> 30.00% 감소

## 결론
직접 그린 테스트 이미지의 원본 크기가 너무 커서 확률이 낮게 나온다고 판단 > 테스트 이미지 원본 크기 94x94로 줄임 > 결과 더 안좋음...