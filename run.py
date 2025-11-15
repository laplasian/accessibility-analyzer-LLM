import json
import numpy as np
from text_feature_extractor import extract_text_features, setup_model_cache
from models import create_text_head_model

def main():
    json_file_path = 'example_dom.json'
    
    try:
        # --- Шаг 1: Настройка и загрузка данных ---
        print("--- Шаг 1: Подготовка ---")
        # Настраиваем кэш для TF Hub, чтобы модель скачалась в папку /models
        setup_model_cache()
        
        print(f"Загрузка данных из '{json_file_path}'...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            dom_data = json.load(f)
        print("Данные успешно загружены.")

        # --- Шаг 2: Извлечение текстовых признаков ---
        print("\n--- Шаг 2: Извлечение текстовых признаков ---")
        # Эта функция вернет матрицу (N, 512), где N - кол-во элементов
        text_vectors = extract_text_features(dom_data)
        
        if text_vectors.shape[0] == 0:
            print("В JSON не найдено элементов для анализа.")
            return
            
        print(f"Признаки извлечены. Форма матрицы: {text_vectors.shape}")

        # --- Шаг 3: Создание и запуск модели ---
        print("\n--- Шаг 3: Создание нейросетевой модели ---")
        # Создаем нашу "голову" для текстовых признаков
        # Указываем корректную форму входа для этой конкретной страницы
        text_model = create_text_head_model(input_shape=(text_vectors.shape[0], text_vectors.shape[1]))
        
        # Модель ожидает на вход "пачку" (batch) данных.
        # Добавляем первую размерность, чтобы (N, 512) превратилось в (1, N, 512).
        model_input = np.expand_dims(text_vectors, axis=0)
        print(f"Входные данные для модели подготовлены. Форма: {model_input.shape}")
        
        print("\n--- Шаг 4: Получение предсказания ---")
        predicted_score = text_model.predict(model_input)
        
        # --- Шаг 5: Вывод результата ---
        print("\n--- Результат ---")
        print(f"Предсказанная оценка юзабилити (от 0 до 100): {predicted_score[0][0]:.2f}")

    except FileNotFoundError:
        print(f"ОШИБКА: Файл '{json_file_path}' не найден. Убедитесь, что он находится в корне проекта.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")

if __name__ == '__main__':
    main()
