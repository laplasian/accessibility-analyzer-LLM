# -*- coding: utf-8 -*-

"""
Модуль для извлечения и обработки числовых (скалярных) признаков
из структурированных JSON-данных, представляющих DOM-дерево.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

# Добавляем корень проекта в sys.path для импорта config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import FEATURE_MAPPING


def _parse_css_value(value):
    """
    Преобразует строковое CSS-значение в число.
    '16px' -> 16.0, 'auto' -> 0.0, '700' -> 700.0
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.lower().strip()
        if value == 'auto':
            return 0.0
        try:
            # Удаляем 'px' и другие возможные единицы, оставляя только число
            return float(''.join(filter(lambda x: x.isdigit() or x == '.', value)))
        except (ValueError, TypeError):
            return 0.0
    return 0.0


def _flatten_and_extract(elements, depth=0):
    """
    Рекурсивно обходит DOM-дерево, извлекая числовые признаки для каждого элемента.
    """
    all_features = []
    numeric_feature_keys = FEATURE_MAPPING.get("numeric_scalar", [])

    for element in elements:
        features = []
        # 1. Извлечение признаков из полей элемента
        for key in numeric_feature_keys:
            if key in ["depth", "num_children"]:
                continue  # Эти признаки вычисляются отдельно
            value = element.get(key)
            features.append(_parse_css_value(value))

        # 2. Вычисление иерархических признаков
        children = element.get("children", [])
        features.append(float(depth))  # depth
        features.append(float(len(children)))  # num_children

        all_features.append(features)

        # 3. Рекурсивный вызов для дочерних элементов
        if children:
            all_features.extend(_flatten_and_extract(children, depth + 1))

    return all_features


def extract_numeric_features(json_data, normalize=True):
    """
    Извлекает, собирает и нормализует числовые признаки из JSON-данных.

    Args:
        json_data (dict): Словарь с данными о DOM-дереве.
        normalize (bool): Если True, выполняет StandardScaler нормализацию.

    Returns:
        np.ndarray: Матрица числовых признаков формы (N, num_features).
    """
    elements = json_data.get("elements", [])
    if not elements:
        return np.array([])

    print("Извлечение числовых признаков...")
    feature_matrix = _flatten_and_extract(elements)
    feature_matrix = np.array(feature_matrix, dtype=np.float32)

    if normalize and feature_matrix.shape[0] > 0:
        print("Нормализация числовых признаков (StandardScaler)...")
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)

    print(f"Матрица числовых признаков создана. Форма: {feature_matrix.shape}")
    return feature_matrix


if __name__ == '__main__':
    import json

    try:
        # Путь к тестовому JSON-файлу
        example_json_path = os.path.join(PROJECT_ROOT, 'dataset', 'sample1.json')

        with open(example_json_path, 'r', encoding='utf-8') as f:
            dom_data = json.load(f)

        numeric_vectors = extract_numeric_features(dom_data, normalize=True)

        print("\nПример первых 5 векторов числовых признаков:")
        print(numeric_vectors[:5])

    except FileNotFoundError:
        print(f"Ошибка: файл '{os.path.basename(example_json_path)}' не найден по пути {example_json_path}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")