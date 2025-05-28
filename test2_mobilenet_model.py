import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# ───────────────────────────────────────────────
# 1. 클래스 및 경로 설정
# ───────────────────────────────────────────────
CLASSES = [
    'whale', 'car', 'tree', 'cat', 'airplane', 'hat', 'dog', 'fish', 'bicycle', 'house',
    'flower', 'star', 'moon', 'clock', 'cloud', 'candle', 'cup', 'book', 'bus', 'camera',
    'chair', 'door', 'guitar', 'hamburger', 'ice cream', 'key', 'laptop', 'pencil', 'shoe', 'cake'
]

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE_DIR = os.path.join(PROJECT_DIR, 'test_images')

# ───────────────────────────────────────────────
# 2. 모델 로딩
# ───────────────────────────────────────────────
mobilenet_model = load_model(os.path.join(PROJECT_DIR, 'mobilenet_model_96.h5'))

# ───────────────────────────────────────────────
# 3. 테스트 및 결과 저장 함수 정의
# ───────────────────────────────────────────────
def test_mobilenet_and_save(model, image_dir=TEST_IMAGE_DIR, output_file="mobilenet_test_results.txt"):
    correct = 0
    total = 0
    results = []

    for file in sorted(os.listdir(image_dir)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            true_class = file.split('.')[0]
            img_path = os.path.join(image_dir, file)

            # 이미지 로딩 및 전처리
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (96, 96))
            img_input = img_resized.astype('float32') / 255.0
            img_input = img_input.reshape(1, 96, 96, 1)

            # MobileNetV2 입력 형식에 맞게 변환
            img_expanded = np.concatenate([img_input] * 3, axis=-1)       # (28,28,3)
            img_upscaled = tf.image.resize(img_expanded, [224, 224])     # (224,224,3)

            # 예측
            pred = model.predict(img_upscaled, verbose=0)
            pred_class = CLASSES[np.argmax(pred)]
            pred_prob = float(np.max(pred))

            is_correct = pred_class == true_class
            correct += int(is_correct)
            total += 1

            # 결과 저장
            result = (
                f"파일: {file}\n"
                f"  → 예측: {pred_class}\n"
                f"  → 일치 여부: {'✅' if is_correct else '❌'}\n"
                f"  → 확률: {pred_prob:.4f}\n"
            )
            results.append(result)

    # 정확도 추가
    accuracy = correct / total
    results.append(f"\n[정확도] MobileNetV2: {accuracy:.2%}\n")

    # 결과 텍스트 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines("\n".join(results))

    print(f"✅ 테스트 결과가 '{output_file}'에 저장되었습니다.")

# ───────────────────────────────────────────────
# 4. 테스트 실행
# ───────────────────────────────────────────────
test_mobilenet_and_save(mobilenet_model)
