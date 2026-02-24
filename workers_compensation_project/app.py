import streamlit as st
from analysis_and_model import analysis_and_model_page
from presentation import presentation_page

st.set_page_config(page_title="Workers Compensation Project")

st.sidebar.title("Навигация")

page = st.sidebar.radio(
    "Выберите страницу:",
    ["Анализ и модель", "Презентация"]
)

if page == "Анализ и модель":
    analysis_and_model_page()
else:
    presentation_page()