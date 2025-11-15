import json
import os
import numpy as np
from text_feature_extractor import extract_text_features

def load_dataset(dataset_dir='dataset', labels_file='labels.json'):
    """
    Загружает датасет: извлекает признаки для каждого JSON-файла и сопоставляет их с оценками.

    Args:
        dataset_dir (str): Путь к папке с данными.
        labels_file (str): Имя файла с метками.

    Returns:
        tuple: Кортеж (X, y), где:
               - X (list): список матриц признаков (каждая матрица для одного JSON).
               - y (np.ndarray): массив оценок.
    """
    labels_path = os.path.join(dataset_dir, labels_file)
    
    print(f"Загрузка меток из файла: {labels_path}")
    try:
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл с метками не найден: {labels_path}")
        return [], np.array([])

    X = []  # Список для хранения матриц признаков
    y = []  # Список для хранения оценок

    print("Обработка файлов данных...")
    for filename, score in labels.items():
        file_path = os.path.join(dataset_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dom_data = json.load(f)
            
            # Извлекаем текстовые признаки для текущего файла
            text_vectors = extract_text_features(dom_data)
            
            if text_vectors.shape[0] > 0:
                X.append(text_vectors)
                y.append(score)
                print(f"  - Файл '{filename}' обработан. Найдено {text_vectors.shape[0]} элементов.")
            else:
                print(f"  - ПРЕДУПРЕЖДЕНИЕ: В файле '{filename}' не найдено элементов. Пропускается.")

        except FileNotFoundError:
            print(f"  - ПРЕДУПРЕЖДЕНИЕ: Файл данных '{filename}' не найден. Пропускается.")
        except Exception as e:
            print(f"  - ОШИБКА при обработке файла '{filename}': {e}. Пропускается.")

    if not X:
        print("Не удалось загрузить ни одного примера данных.")
        return [], np.array([])

    return X, np.array(y)

if __name__ == '__main__':
    # Пример использования
    from text_feature_extractor import setup_model_cache

    print("--- Демонстрация загрузчика данных ---")
    # Настраиваем кэш перед использованием extract_text_features
    setup_model_cache()
    
    # Загружаем датасет
    features, scores = load_dataset()
    
    if features:
        print("\n--- Результат загрузки ---")
        print(f"Загружено {len(features)} примеров.")
        print(f"Оценки (y): {scores}")
        print("Признаковые матрицы (X):")
        for i, matrix in enumerate(features):
            print(f"  - Пример {i+1}: форма матрицы {matrix.shape}")
