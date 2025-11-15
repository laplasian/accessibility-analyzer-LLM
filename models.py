import tensorflow as tf
from tensorflow.keras import layers
import config



# --- Входной слой ---
# (batch_size, N, F)
inputs = layers.Input(shape=(sequence_length, feature_size))

# --- Маскирование (для padding) ---
mask = layers.Masking(mask_value=0.0)(inputs)

# --- Блок Трансформера-Энкодера ---
# Внимание "смотрит" на все элементы и находит связи
attn_output = layers.MultiHeadAttention(
    num_heads=num_heads,
    key_dim=transformer_units
)(mask, mask)
x = layers.LayerNormalization()(mask + attn_output) # Residual connection
x = layers.Dense(transformer_units, activation="relu")(x) # FeedForward
# (Этот блок можно повторить 1-3 раза)

# --- Агрегация (Свертка) ---
# "Схлопываем" N векторов элементов в 1 вектор,
# представляющий всю страницу
page_vector = layers.GlobalAveragePooling1D()(x)

# --- "Голова" Классификатора ---
x = layers.Dense(64, activation="relu")(page_vector)
x = layers.Dropout(0.3)(x)

# --- Выходной слой (Регрессия) ---
# Предсказываем одно число (оценку)
outputs = layers.Dense(1, activation="linear")(x)

# --- Сборка модели ---
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer="adam",
    loss="mean_squared_error", # MSE для регрессии
    metrics=["mean_absolute_error"] # MAE
)

model.summary()