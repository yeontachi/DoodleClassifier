import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, GlobalAveragePooling2D, Concatenate)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ────────────── 1. 설정 ──────────────
CLASSES = [  # 클래스 목록 (30개)
    'whale', 'car', 'tree', 'cat', 'airplane', 'hat', 'dog', 'fish', 'bicycle', 'house',
    'flower', 'star', 'moon', 'clock', 'cloud', 'candle', 'cup', 'book', 'bus', 'camera',
    'chair', 'door', 'guitar', 'hamburger', 'ice cream', 'key', 'laptop', 'pencil', 'shoe', 'cake'
]
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_DIR, 'quickdraw_data')

# ────────────── 2. 데이터 로드 ──────────────
def load_and_preprocess_data(classes, samples_per_class=3000):
    X, y = [], []
    for i, cls in enumerate(classes):
        try:
            data = np.load(os.path.join(DATA_PATH, f"{cls}.npy"))
            np.random.shuffle(data)
            data = data[:samples_per_class]
            X.extend(data)
            y.extend([i] * len(data))
        except Exception as e:
            print(f"Error loading {cls}: {e}")
    X = np.array(X).astype('float32') / 255.0
    X = X.reshape(-1, 28, 28, 1)
    X = tf.image.resize(X, [96, 96]).numpy()
    y = np.array(y)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]

# ────────────── 3. 모델 정의 ──────────────
def create_mobilenet_model(num_classes):
    input_layer = Input(shape=(96, 96, 1))
    x = Concatenate()([input_layer, input_layer, input_layer])  # (96,96,3)로 복제
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=x)
    base_model.trainable = True  # Fine-tuning
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ────────────── 4. 데이터 준비 ──────────────
X, y = load_and_preprocess_data(CLASSES)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_train = to_categorical(y_train, len(CLASSES))
y_test = to_categorical(y_test, len(CLASSES))

# ────────────── 5. 학습 설정 ──────────────
class_weights = compute_class_weight('balanced', classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1))
class_weight_dict = dict(enumerate(class_weights))

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
]

datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_gen = datagen.flow(X_train, y_train, batch_size=128, subset='training')
val_gen = datagen.flow(X_train, y_train, batch_size=128, subset='validation')

# ────────────── 6. 학습 ──────────────
mobilenet_model = create_mobilenet_model(len(CLASSES))
mobilenet_model.fit(train_gen, epochs=30, validation_data=val_gen, callbacks=callbacks, class_weight=class_weight_dict)

# ────────────── 7. 저장 ──────────────
mobilenet_model.save(os.path.join(PROJECT_DIR, 'mobilenet_model_96_finetuned.h5'))
print("✅ MobileNetV2 fine-tuned model saved.")
