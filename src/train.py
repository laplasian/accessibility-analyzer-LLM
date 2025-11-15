import numpy as np
from tensorflow import keras
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data_loader import load_dataset
from src.models import create_text_head_model
from src.text_feature_extractor import setup_model_cache

def train_model(epochs=5, batch_size=1):
    # --- Шаг 1: Подготовка и загрузка данных ---
    setup_model_cache()
    X_train, y_train = load_dataset()

    if len(X_train) == 0:
        return

    if batch_size > 1:
        batch_size = 1
        
    # --- Шаг 2: Создание модели ---
    model = create_text_head_model(input_shape=(None, 512))
    model.summary()

    # --- Шаг 3: Обучение модели ---
    print(f"Количество эпох: {epochs}")
    print(f"Размер пакета (batch size): {batch_size}")
    
    for epoch in range(epochs):
        print(f"\nЭпоха {epoch + 1}/{epochs}")
        total_loss = 0
        total_mae = 0
        for i, (x_sample, y_sample) in enumerate(zip(X_train, y_train)):
            x_sample_batch = np.expand_dims(x_sample, axis=0)
            y_sample_batch = np.array([y_sample])
            
            results = model.train_on_batch(x_sample_batch, y_sample_batch)
            
            loss = results[0]
            mae = results[1]
            
            total_loss += loss
            total_mae += mae
            
            print(f"  Пример {i+1}/{len(X_train)} обработан. Loss: {loss:.4f}, MAE: {mae:.4f}")
        
        avg_loss = total_loss / len(X_train)
        avg_mae = total_mae / len(X_train)
        print(f"Средние метрики за эпоху: Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}")

    # --- Шаг 4: Сохранение модели ---
    model_save_path = os.path.join(PROJECT_ROOT, 'trained_model.keras')
    model.save(model_save_path)
    print(f"Модель успешно сохранена в файл: {model_save_path}")


if __name__ == '__main__':
    train_model(epochs=1000)
