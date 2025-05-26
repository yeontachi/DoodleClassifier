import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns
from tqdm import tqdm
from google.colab import files

CLASSES = [
    'whale', 'car', 'tree', 'cat', 'airplane', 'hat', 'dog', 'fish', 'bicycle', 'house',
    'flower', 'star', 'moon', 'clock', 'cloud', 'candle', 'cup', 'book', 'bus', 'camera',
    'chair', 'door', 'guitar', 'hamburger', 'ice cream', 'key', 'laptop', 'pencil', 'shoe', 'cake'
]

DATA_PATH = '/content/quickdraw_data'
os.makedirs(DATA_PATH, exist_ok=True)

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

    X = np.array(X)
    y = np.array(y)

    X = X.astype('float32') / 255.0
    X = X.reshape(-1, 28, 28, 1)

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    return X, y

X, y = load_and_preprocess_data(CLASSES)
print(f"Data shape: {X.shape}")
print(f"Label shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_train = to_categorical(y_train, len(CLASSES))
y_test = to_categorical(y_test, len(CLASSES))

print(f"Train data: {X_train.shape}")
print(f"Test data: {X_test.shape}")
