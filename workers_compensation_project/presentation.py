import streamlit as st
import reveal_slides as rs
def presentation_page():

    st.title("Презентация проекта")

    presentation_markdown = """
# Прогнозирование страховых выплат

---

## Бизнес-задача
Предсказать итоговую стоимость страхового возмещения.

---

## Датасет
Workers Compensation (100 000 записей)

---

## Модели
- Linear Regression
- Random Forest
- XGBoost
- Ridge

---

## Метрики
- MAE
- RMSE
- R²

---

## Результаты
Лучшая модель показала высокий R².

---

## Итог
Модель помогает страховым компаниям точнее формировать резервы.
"""

    rs.slides(presentation_markdown, height=500)