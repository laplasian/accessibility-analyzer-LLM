import json
import numpy as np
import os
from src.feature_extractor.text_feature_extractor import extract_text_features, setup_model_cache
from src.models import create_text_head_model

def main():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(PROJECT_ROOT, 'dataset', 'example_dom.json')
    
    try:
        setup_model_cache()
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            dom_data = json.load(f)

        text_vectors = extract_text_features(dom_data)
        
        if text_vectors.shape[0] == 0:
            return
            
        text_model = create_text_head_model(input_shape=(text_vectors.shape[0], text_vectors.shape[1]))
        
        model_input = np.expand_dims(text_vectors, axis=0)
        
        predicted_score = text_model.predict(model_input)
        
    except FileNotFoundError:
        pass
    except Exception as e:
        pass

if __name__ == '__main__':
    main()
