import numpy as np
from tensorflow import keras

# Импортируем наши модули
from data_loader import load_dataset
from models import create_text_head_model
from text_feature_extractor import setup_model_cache

def train_model(epochs=5, batch_size=1):
    """
    Основной сценарий обучения модели.
    """
    print("--- Начало процесса обучения ---")

    # --- Шаг 1: Подготовка и загрузка данных ---
    print("\n--- Шаг 1: Загрузка датасета ---")
    # Настраиваем кэш для TF Hub, чтобы модель скачалась/загрузилась корректно
    setup_model_cache()
    # Загружаем данные (признаки и метки)
    X_train, y_train = load_dataset()

    if len(X_train) == 0:
        print("\nОбучение невозможно: данные не загружены.")
        return

    # ВАЖНО: Модели Keras требуют, чтобы все входные данные в одном пакете (batch)
    # имели одинаковую форму. У нас страницы с разным количеством элементов.
    # Самый простой способ это решить - обучать с batch_size=1.
    # Для больших датасетов используют padding (дополнение нулями).
    if batch_size > 1:
        print("\nПРЕДУПРЕЖДЕНИЕ: Установлен batch_size > 1, но данные не выровнены (padding).")
        print("Для демонстрации будет использоваться batch_size = 1.")
        batch_size = 1
        
    # --- Шаг 2: Создание модели ---
    print("\n--- Шаг 2: Создание модели ---")
    # Мы не можем заранее знать input_shape, т.к. страницы разные.
    # Поэтому мы создаем модель с `input_shape=(None, 512)`.
    # `None` означает, что количество элементов может быть любым.
    model = create_text_head_model(input_shape=(None, 512))
    print("Архитектура модели:")
    model.summary()

    # --- Шаг 3: Обучение модели ---
    print("\n--- Шаг 3: Запуск обучения ---")
    print(f"Количество эпох: {epochs}")
    print(f"Размер пакета (batch size): {batch_size}")
    
    # Так как у нас batch_size=1, мы можем обучать модель в простом цикле.
    # Для реальных задач используют tf.data.Dataset.
    for epoch in range(epochs):
        print(f"\nЭпоха {epoch + 1}/{epochs}")
        total_loss = 0
        for i, (x_sample, y_sample) in enumerate(zip(X_train, y_train)):
            # Добавляем batch-размерность: (N, 512) -> (1, N, 512)
            x_sample_batch = np.expand_dims(x_sample, axis=0)
            y_sample_batch = np.array([y_sample])
            
            # Один шаг обучения
            loss = model.train_on_batch(x_sample_batch, y_sample_batch)
            total_loss += loss
            print(f"  Пример {i+1}/{len(X_train)} обработан. Loss: {loss:.4f}")
        
        print(f"Средняя ошибка (loss) за эпоху: {total_loss / len(X_train):.4f}")

    # --- Шаг 4: Сохранение модели ---
    print("\n--- Шаг 4: Сохранение обученной модели ---")
    model_save_path = 'trained_model.h5'
    model.save(model_save_path)
    print(f"Модель успешно сохранена в файл: {model_save_path}")


if __name__ == '__main__':
    # Запускаем обучение с небольшим количеством эпох для демонстрации
    train_model(epochs=3)
