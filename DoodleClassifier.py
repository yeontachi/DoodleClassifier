import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import urllib.request
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     GlobalAveragePooling2D, BatchNormalization, Input,
                                     Concatenate, UpSampling2D)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ───────────────────────────────────────────────
# 1. 클래스 및 데이터 경로 설정
# ───────────────────────────────────────────────
CLASSES = [
    'whale', 'car', 'tree', 'cat', 'airplane', 'hat', 'dog', 'fish', 'bicycle', 'house',
    'flower', 'star', 'moon', 'clock', 'cloud', 'candle', 'cup', 'book', 'bus', 'camera',
    'chair', 'door', 'guitar', 'hamburger', 'ice cream', 'key', 'laptop', 'pencil', 'shoe', 'cake'
]

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, 'quickdraw_data')
TEST_IMAGE_DIR = os.path.join(PROJECT_DIR, 'test_images')

os.makedirs(DATA_PATH, exist_ok=True)

# ───────────────────────────────────────────────
# 2. 데이터 다운로드
# ───────────────────────────────────────────────
def download_quickdraw_data(classes):
    for cls in tqdm(classes, desc="Downloading"):
        filename = cls.replace(' ', '%20')
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{filename}.npy"
        output_path = os.path.join(DATA_PATH, f"{cls}.npy")

        if not os.path.exists(output_path):
            try:
                urllib.request.urlretrieve(url, output_path)
            except Exception as e:
                print(f"Failed {cls}: {e}")

download_quickdraw_data(CLASSES)

# ───────────────────────────────────────────────
# 3. 데이터 로딩 및 전처리
# ───────────────────────────────────────────────
def load_and_preprocess_data(classes, samples_per_class=2000):
    X, y = [], []

    for i, cls in enumerate(tqdm(classes, desc="Loading data")):
        try:
            data = np.load(os.path.join(DATA_PATH, f"{cls}.npy"))
            np.random.shuffle(data)
            data = data[:samples_per_class]
            X.extend(data)
            y.extend([i] * len(data))
        except Exception as e:
            print(f"Loading failed {cls}: {e}")

    X = np.array(X).astype('float32') / 255.0
    X = X.reshape(-1, 28, 28, 1)
    y = np.array(y)

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]

X, y = load_and_preprocess_data(CLASSES)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

y_train = to_categorical(y_train, len(CLASSES))
y_test = to_categorical(y_test, len(CLASSES))

# ───────────────────────────────────────────────
# 4. 모델 정의
# ───────────────────────────────────────────────
def create_basic_cnn(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_mobilenet_model(num_classes):
    input_layer = Input(shape=(28, 28, 1))
    x = Concatenate()([input_layer, input_layer, input_layer])  # (28,28,3)
    x = UpSampling2D(size=(8,8))(x)  # (224,224,3)

    base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

basic_model = create_basic_cnn(len(CLASSES))
mobilenet_model = create_mobilenet_model(len(CLASSES))
