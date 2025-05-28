## 테스트 결과 요약 보고서
 - 총 테스트 이미지 : 30장
 - 이미지 경로: test_images(60x60)/
 - 이미지 파일명: 각 클래스 이름과 동일(예: airplane.png, car.png 등)
 - 테스트 이미지 30장 실제 크기 60x60로 줄임

 ### Basic CNN model(basic_cnn_model_96.h5)
 - 일치 : 1개
 - 불일치 : 29개
 - 정확도 : 3.33%
 - 예측 결과 이미지 > test4_basicCNN_model.png

### MobileNetV2 model(mobilenet_model_96.h5)
 - 일치 : 14개
 - 불일치 : 16개
 - 정확도 : 46.67%
 - 예측 결과 이미지 > test4_mobilenet_model.png

## 이전 테스트(test1) 결과와 비교 
- Basic CNN: 3.33% -> 3.33% 동일
- MobileNetV2: 36.67% -> 46.67% 증가

## 결론
테스트 이미지를 60x60 크기로 변경 > 전이학습한 mobilenet 모델의 경우 정확도 46.67%까지 증가
이전에 테스트 이미지를 줄였을 때랑 다르게 확률이 확연하게 차이남(증가가)