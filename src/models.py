import tensorflow as tf
import keras
from keras import layers
import numpy as np

def create_text_head_model(input_shape=(None, 512), num_heads=8, ff_dim=512, num_transformer_blocks=1):

    inputs = layers.Input(shape=input_shape, name="text_input")

    # --- Блок Трансформера ---
    x = inputs
    for _ in range(num_transformer_blocks):
        # Слой внимания (Multi-Head Attention)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1] // num_heads)(x, x)
        # Остаточное соединение и нормализация
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        # Полносвязная сеть (Feed-Forward)
        ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(input_shape[-1])]
        )
        ffn_output = ffn(x)
        # Остаточное соединение и нормализация
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

    # --- Агрегация ---
    # Слой GlobalAveragePooling1D "схлопывает" информацию обо всех N элементах в один вектор,
    # представляющий всю страницу.
    x = layers.GlobalAveragePooling1D(name="page_vector")(x)

    # --- Классификатор ("Голова") ---
    # Несколько полносвязных слоев для финального предсказания
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu", name="dense_2")(x)
    
    # Выходной слой: один нейрон для регрессии (оценка от 0 до 100)
    outputs = layers.Dense(1, activation="linear", name="usability_score")(x)

    # Создание и компиляция модели
    model = keras.Model(inputs=inputs, outputs=outputs, name="text_head_model")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="mean_squared_error",  # MSE - стандартная функция потерь для регрессии
        metrics=["mean_absolute_error"] # MAE - показывает, на сколько в среднем ошибается модель
    )
    return model