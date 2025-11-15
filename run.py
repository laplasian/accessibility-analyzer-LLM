import json
import numpy as np
import os
import sys

# --- НАСТРОЙКА ПУТЕЙ ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --- КОНЕЦ НАСТРОЙКИ ПУТЕЙ ---

from src.text_feature_extractor import extract_text_features, setup_model_cache
from src.models import create_text_head_model

def main():
    """
    Основной сценарий:
    1. Загружает данные из JSON.
    2. Извлекает текстовые признаки.
    3. Создает модель.
    4. Делает предсказание.
    """
    json_file_path = os.path.join(PROJECT_ROOT, 'dataset', 'example_dom.json')
    
    try:
        # --- Шаг 1: Подготовка и загрузка данных ---
        print("--- Шаг 1: Подготовка ---")
        setup_model_cache()
        
        print(f"Загрузка данных из '{json_file_path}'...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            dom_data = json.load(f)
        print("Данные успешно загружены.")

        # --- Шаг 2: Извлечение текстовых признаков ---
        print("\n--- Шаг 2: Извлечение текстовых признаков ---")
        text_vectors = extract_text_features(dom_data)
        
        if text_vectors.shape[0] == 0:
            print("В JSON не найдено элементов для анализа.")
            return
            
        print(f"Признаки извлечены. Форма матрицы: {text_vectors.shape}")

        # --- Шаг 3: Создание и запуск модели ---
        print("\n--- Шаг 3: Создание нейросетевой модели ---")
        text_model = create_text_head_model(input_shape=(text_vectors.shape[0], text_vectors.shape[1]))
        
        model_input = np.expand_dims(text_vectors, axis=0)
        print(f"Входные данные для модели подготовлены. Форма: {model_input.shape}")
        
        print("\n--- Шаг 4: Получение предсказания ---")
        predicted_score = text_model.predict(model_input)
        
        # --- Шаг 5: Вывод результата ---
        print("\n--- Результат ---")
        print(f"Предсказанная оценка юзабилити (от 0 до 100): {predicted_score[0][0]:.2f}")

    except FileNotFoundError:
        print(f"ОШИБКА: Файл '{json_file_path}' не найден.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")

if __name__ == '__main__':
    main()
