import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

def create_text_head_model(input_shape=(None, 512), num_heads=8, ff_dim=512, num_transformer_blocks=1):
    """
    Создает модель "головы" для обработки текстовых признаков.
    """
    inputs = layers.Input(shape=input_shape, name="text_input")

    x = inputs
    for _ in range(num_transformer_blocks):
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1] // num_heads)(x, x)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(input_shape[-1])]
        )
        ffn_output = ffn(x)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    x = layers.GlobalAveragePooling1D(name="page_vector")(x)

    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    
    outputs = layers.Dense(1, activation="linear", name="usability_score")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="text_head_model")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )
    
    return model

if __name__ == '__main__':
    text_model = create_text_head_model()
    
    print("Архитектура модели для текстовых признаков:")
    text_model.summary()
    
    dummy_page_vectors = np.random.rand(1, 25, 512) 
    
    print(f"\nПример входных данных: 1 страница, {dummy_page_vectors.shape[1]} элементов, {dummy_page_vectors.shape[2]} признаков в каждом.")
    
    predicted_score = text_model.predict(dummy_page_vectors)
    
    print(f"\nПример предсказания (оценка юзабилити): {predicted_score[0][0]:.2f}")
