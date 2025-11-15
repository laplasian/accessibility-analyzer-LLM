```bash
 conda init
 conda env create -f env.yml
 conda activate env
 
```

Для каждого элемента `element` в списке `elements`:

1.  **Текстовые признаки:**
    * **Поля:** `text`, `alt`, `title`, `placeholder`.
    * **Процесс:** Конкатенировать (склеить) текст из этих полей.
    * **Векторизация:** Подать полученную строку в предобученную модель `Universal Sentence Encoder` (из `tensorflow_hub`).
    * **Выход:** `vector_text` (размер **512**).

2.  **Числовые (скалярные) признаки:**
    * **Поля (Геометрия):** `width`, `height`, `top`, `left`. (Распарсить 'px', 'auto' -> 0).
    * **Поля (Стиль):** `fontSize`, `fontWeight`, `lineHeight`, `opacity`, `letterSpacing`.
    * **Поля (Иерархия):** `depth` (уровень вложенности, вычисляется рекурсивно), `num_children` (длина списка `children`).
    * **Процесс:** Собрать все эти числа в один вектор и **нормализовать** (например, `StandardScaler` или `MinMaxScaler`).
    * **Выход:** `vector_numeric` (размер ~**15-20**).

3.  **Цветовые признаки:**
    * **Поля:** `color`, `backgroundColor`.
    * **Процесс:** Преобразовать `rgb(r, g, b)` в `[R/255, G/255, B/255]`.
    * **Доп. признак:** Рассчитать **коэффициент контрастности WCAG** между `color` и `backgroundColor`. Это число (от 1 до 21) добавить в `vector_numeric`.
    * **Выход:** `vector_colors` (размер $3 + 3 = 6$).

4.  **Категориальные признаки:**
    * **Поля:** `tag` ('div', 'a', 'p'...), `position` ('static', 'relative'...), `display` ('block', 'inline'...), `textAlign` ('left', 'center'...).
    * **Процесс:** Для каждого поля создается словарь (e.g., `'div': 1, 'a': 2...`). Эти числовые индексы будут подаваться в `Embedding`-слои внутри Keras.
