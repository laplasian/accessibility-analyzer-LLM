# -*- coding: utf-8 -*-

"""
Модуль для извлечения и обработки цветовых признаков (color, backgroundColor)
и расчета контрастности WCAG.
"""

import numpy as np
import re


def _parse_rgb(color_str):
    """Парсит 'rgb(r, g, b)' или 'rgba(r, g, b, a)' в список [r, g, b]."""
    if not isinstance(color_str, str):
        return [0, 0, 0]  # По умолчанию черный

    match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color_str.lower())
    if match:
        return [int(c) for c in match.groups()]

    match_rgba = re.search(r'rgba\((\d+),\s*(\d+),\s*(\d+),.*?\)', color_str.lower())
    if match_rgba:
        return [int(c) for c in match_rgba.groups()]

    # Можно добавить обработку hex, 'white', 'black' и т.д.
    # Для простоты пока оставляем только rgb
    if color_str == 'transparent':
        return [0, 0, 0]  # Не совсем корректно, но нужно для расчета

    return [0, 0, 0]  # По умолчанию черный


def _get_luminance(r, g, b):
    """Рассчитывает относительную яркость (luminance) для sRGB."""
    # Нормализация [0, 255] -> [0, 1]
    rgb = np.array([r, g, b]) / 255.0

    # Применение гамма-коррекции
    rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)

    # Формула яркости WCAG
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _get_contrast_ratio(rgb1, rgb2):
    """Рассчитывает коэффициент контрастности WCAG между двумя RGB-цветами."""
    try:
        lum1 = _get_luminance(*rgb1)
        lum2 = _get_luminance(*rgb2)

        if lum1 > lum2:
            return (lum1 + 0.05) / (lum2 + 0.05)
        else:
            return (lum2 + 0.05) / (lum1 + 0.05)
    except Exception:
        return 1.0  # Минимальный контраст в случае ошибки


def _flatten_and_extract_colors(elements):
    """Рекурсивно обходит DOM, извлекая цветовые признаки."""
    all_color_features = []
    all_contrast_features = []

    for element in elements:
        color_str = element.get("color")
        bg_color_str = element.get("backgroundColor")

        rgb_color = _parse_rgb(color_str)
        rgb_bg_color = _parse_rgb(bg_color_str)

        # 1. Признаки цветов (нормализованные)
        features = (np.array(rgb_color + rgb_bg_color) / 255.0).tolist()
        all_color_features.append(features)

        # 2. Признак контрастности
        contrast = _get_contrast_ratio(rgb_color, rgb_bg_color)
        all_contrast_features.append([contrast])  # Как отдельный признак

        # 3. Рекурсивный вызов для дочерних элементов
        children = element.get("children", [])
        if children:
            child_colors, child_contrasts = _flatten_and_extract_colors(children)
            all_color_features.extend(child_colors)
            all_contrast_features.extend(child_contrasts)

    return all_color_features, all_contrast_features


def extract_color_features(json_data):
    """
    Извлекает цветовые признаки (6 штук) и признак контрастности (1 штука).

    Returns:
        tuple: (
            np.ndarray: Матрица цветовых признаков (N, 6),
            np.ndarray: Матрица контрастности (N, 1) - пойдет в числовые признаки
        )
    """
    elements = json_data.get("elements", [])
    if not elements:
        return np.array([]), np.array([])

    print("Извлечение цветовых признаков...")
    color_matrix, contrast_matrix = _flatten_and_extract_colors(elements)

    color_matrix = np.array(color_matrix, dtype=np.float32)
    contrast_matrix = np.array(contrast_matrix, dtype=np.float32)

    print(f"Матрица цветовых признаков создана. Форма: {color_matrix.shape}")
    print(f"Матрица контрастности создана. Форма: {contrast_matrix.shape}")

    return color_matrix, contrast_matrix