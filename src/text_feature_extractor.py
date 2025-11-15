import json
import tensorflow_hub as hub
import numpy as np
import os

MODELS_DIR = 'models'
MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
VECTOR_SIZE = 512

def setup_model_cache():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.environ['TFHUB_CACHE_DIR'] = MODELS_DIR
    print(f"Кэш моделей TensorFlow Hub будет находиться в папке: {os.path.abspath(MODELS_DIR)}")

def extract_text_features(json_data):
    elements = json_data.get("elements", [])
    if not elements:
        return np.array([])

    # --- Загрузка модели ---
    print("\nЗагрузка модели Universal Sentence Encoder...")
    print(f"URL: {MODEL_URL}")
    print("При первом запуске модель будет скачана и сохранена в папку 'models'. Это может занять несколько минут.")
    
    embed = hub.load(MODEL_URL)
    
    print("Модель успешно загружена.")
    # ---------------------

    text_features = []
    for element in elements:
        text_parts = [
            element.get("text", ""),
            element.get("alt", ""),
            element.get("title", ""),
            element.get("placeholder", "")
        ]
        concatenated_text = " ".join(filter(None, text_parts)).strip()

        if concatenated_text:
            vector = embed([concatenated_text])[0].numpy()
        else:
            vector = np.zeros(VECTOR_SIZE)
        
        text_features.append(vector)

    return np.array(text_features)

