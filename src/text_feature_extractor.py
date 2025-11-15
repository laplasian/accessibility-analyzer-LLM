import tensorflow_hub as hub
import numpy as np
import os

MODELS_DIR = 'models'
MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
VECTOR_SIZE = 512

def setup_model_cache():
    """Настраивает папку для кэширования моделей TensorFlow Hub в корне проекта."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(base_dir, MODELS_DIR)
    
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TFHUB_CACHE_DIR'] = cache_dir
    print(f"Кэш моделей TensorFlow Hub будет находиться в папке: {os.path.abspath(cache_dir)}")

def extract_text_features(json_data):
    """
    Извлекает и векторизует текстовые признаки из JSON-данных.
    """
    elements = json_data.get("elements", [])
    if not elements:
        return np.array([])

    print("\nЗагрузка модели Universal Sentence Encoder...")
    print(f"URL: {MODEL_URL}")
    
    embed = hub.load(MODEL_URL)
    
    print("Модель успешно загружена.")

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

if __name__ == '__main__':
    import json
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        setup_model_cache()
        
        example_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'example_dom.json')
        
        with open(example_json_path, 'r', encoding='utf-8') as f:
            dom_data = json.load(f)
        
        text_vectors = extract_text_features(dom_data)
        
        print("\nМатрица текстовых признаков успешно создана.")
        print(f"Форма матрицы: {text_vectors.shape}")

    except FileNotFoundError:
        print(f"Ошибка: файл 'example_dom.json' не найден.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
