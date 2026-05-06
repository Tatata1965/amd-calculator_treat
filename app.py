import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Калькулятор анатомического ответа на анти‑VEGF", layout="wide")
st.title("🔬 Калькулятор прогноза анатомического ответа на анти‑VEGF терапию")

@st.cache_resource
def load_model():
    model = joblib.load('rf_6features_анатомический.pkl')
    scaler = joblib.load('scaler_6features.pkl')
    encoder = joblib.load('label_encoder_6features.pkl')
    return model, scaler, encoder

model, scaler, encoder = load_model()
feature_names = model.feature_names_in_

st.markdown("### Введите данные пациента")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Возраст (лет)", 50, 100, 72)
    cts = st.number_input("ЦТС_до (мкм)", 50, 800, 300)
    mkoz = st.number_input("МКОЗ_до", 0.01, 1.0, 0.5, 0.05)
with col2:
    drug = st.selectbox("Препарат", ["Эйлеа", "Визкью", "Авастин"])
    inj = st.selectbox("Количество инъекций в курсе", [1, 2, 3])
    interval = st.slider("Интервал до контроля (дни)", 20, 60, 40)

if st.button("Прогноз"):
    # Создаём DataFrame с исходными данными
    df = pd.DataFrame({
        'возраст': [age],
        'ЦТС_до': [cts],
        'МКОЗ_до': [mkoz],
        'препарат': [drug],
        'количество_инъекций': [inj],
        'интервал_дней': [interval]
    })
    # One-hot encoding для препарата
    df = pd.get_dummies(df, columns=['препарат'], drop_first=True)
    # Убеждаемся, что есть все колонки для базовой категории (если нет, добавляем)
    # Теперь приводим к полному набору признаков (feature_names)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    # Переставляем колонки в нужном порядке
    df = df[feature_names]
    # Масштабируем числовые колонки (они же должны быть в df)
    numeric_cols = ['возраст', 'ЦТС_до', 'МКОЗ_до', 'количество_инъекций', 'интервал_дней']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    # Предсказание
    probs = model.predict_proba(df)[0]
    pred_class = np.argmax(probs)
    pred_name = encoder.inverse_transform([pred_class])[0]
    classes = encoder.classes_
    prob_dict = dict(zip(classes, probs))
    st.markdown("---")
    st.subheader("📊 Результаты прогноза")
    st.markdown("**Анатомический исход (изменение отёка):**")
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    col_a1.metric("Значительное уменьшение", f"{prob_dict.get('значительное_уменьшение', 0)*100:.1f}%")
    col_a2.metric("Умеренное уменьшение", f"{prob_dict.get('умеренное_уменьшение', 0)*100:.1f}%")
    col_a3.metric("Стабилизация", f"{prob_dict.get('стабилизация', 0)*100:.1f}%")
    col_a4.metric("Увеличение", f"{prob_dict.get('увеличение', 0)*100:.1f}%")
    st.markdown(f"**Наиболее вероятный исход:** {pred_name}")
    st.caption("Модель Random Forest обучена на 6 клинических параметрах. Точность прогноза – около 71%.")
