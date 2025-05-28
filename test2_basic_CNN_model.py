import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
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
# 2. 모델 로딩 (이미 학습된 모델이 저장되어 있어야 함)
# ───────────────────────────────────────────────
basic_model = load_model(os.path.join(PROJECT_DIR, 'basic_cnn_model_96.h5'))
mobilenet_model = load_model(os.path.join(PROJECT_DIR, 'mobilenet_model_96.h5'))

# ───────────────────────────────────────────────
# 3. 테스트 및 시각화 함수 정의
# ───────────────────────────────────────────────
def test_and_visualize_images(model, model_name, image_dir=TEST_IMAGE_DIR):
    correct = 0
    total = 0

    plt.figure(figsize=(15, 30))
    idx = 1

    for file in sorted(os.listdir(image_dir)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            true_class = file.split('.')[0]
            img_path = os.path.join(image_dir, file)

            # 이미지 로딩 및 전처리
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img, (96, 96))
            img_input = img_resized.astype('float32') / 255.0
            img_input = img_input.reshape(1, 96, 96, 1)

            # 예측
            pred = model.predict(img_input, verbose=0)
            pred_class = CLASSES[np.argmax(pred)]
            pred_prob = np.max(pred)

            # 정답 비교
            is_correct = pred_class == true_class
            total += 1
            correct += int(is_correct)

            # 시각화
            plt.subplot(10, 3, idx)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            title_color = 'green' if is_correct else 'red'
            plt.title(f"{file}\n→ {pred_class} ({pred_prob:.2f})", color=title_color, fontsize=10)
            idx += 1

            # 콘솔 출력
            print(f"[{model_name}] File: {file}")
            print(f"  → Predicted: {pred_class}")
            print(f"  → Match: {'✅' if is_correct else '❌'}")
            print(f"  → Probability: {pred_prob:.4f}\n")

    acc = correct / total
    print(f"\n✅ [{model_name}] Accuracy on test images: {acc:.2%}")
    plt.tight_layout()
    plt.show()

# ───────────────────────────────────────────────
# 4. 두 모델로 테스트 수행
# ───────────────────────────────────────────────
test_and_visualize_images(basic_model, "Basic CNN")
test_and_visualize_images(mobilenet_model, "MobileNetV2")
