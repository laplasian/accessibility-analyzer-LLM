# -*- coding: utf-8 -*-

"""
Модуль для извлечения категориальных признаков.
Преобразует строковые значения ('div', 'block') в числовые индексы
для последующей подачи в Embedding-слои.
"""

import numpy as np
import os
import sys

# Добавляем корень проекта в sys.path для импорта config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import FEATURE_MAPPING, CATEGORICAL_VOCABULARIES


def _build_vocab_map(vocab_list):
    """
    Создает словарь {'value': index} из списка.
    Индекс 0 зарезервирован для 'UNK' (неизвестное значение).
    """
    return {value: i + 1 for i, value in enumerate(vocab_list)}


# Создаем словари для всех категориальных признаков
VOCAB_MAPS = {
    key: _build_vocab_map(CATEGORICAL_VOCABULARIES[key])
    for key in FEATURE_MAPPING.get("categorical", [])
}
# Индекс для 'UNK' (Unknown)
UNK_INDEX = 0


def _flatten_and_extract_cats(elements):
    """
    Рекурсивно обходит DOM-дерево, извлекая категориальные признаки (индексы).
    """
    all_features = []
    categorical_keys = FEATURE_MAPPING.get("categorical", [])

    for element in elements:
        features = {}
        for key in categorical_keys:
            value = str(element.get(key, "")).lower().strip()
            # Получаем индекс из словаря или UNK_INDEX, если значение не найдено
            features[key] = VOCAB_MAPS[key].get(value, UNK_INDEX)

        all_features.append(features)

        # Рекурсивный вызов для дочерних элементов
        children = element.get("children", [])
        if children:
            all_features.extend(_flatten_and_extract_cats(children))

    return all_features


def extract_categorical_features(json_data):
    """
    Извлекает и индексирует категориальные признаки из JSON-данных.

    Args:
        json_data (dict): Словарь с данными о DOM-дереве.

    Returns:
        dict: Словарь, где ключ - имя признака (e.g., 'tag'),
              а значение - np.ndarray (N, 1) с индексами.
    """
    elements = json_data.get("elements", [])
    if not elements:
        return {}

    print("Извлечение категориальных признаков (индексация)...")

    flat_features_list = _flatten_and_extract_cats(elements)

    # Транспонирование списка словарей в словарь списков
    feature_dict = {key: [] for key in FEATURE_MAPPING.get("categorical", [])}
    for item in flat_features_list:
        for key in feature_dict.keys():
            feature_dict[key].append(item[key])

    # Преобразование в numpy-массивы (N, 1)
    for key in feature_dict.keys():
        feature_dict[key] = np.array(feature_dict[key], dtype=np.int32).reshape(-1, 1)

    print("Словари категориальных признаков созданы.")
    return feature_dict