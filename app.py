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
    input_data = pd.DataFrame({
        'возраст': [age],
        'ЦТС_до': [cts],
        'МКОЗ_до': [mkoz],
        'препарат': [drug],
        'количество_инъекций': [inj],
        'интервал_дней': [interval]
    })
    input_data = pd.get_dummies(input_data, columns=['препарат'], drop_first=True)
    expected_cols = ['возраст', 'ЦТС_до', 'МКОЗ_до', 'количество_инъекций', 'интервал_дней',
                     'препарат_Визкью', 'препарат_Авастин']
    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_cols]
    num_cols = ['возраст', 'ЦТС_до', 'МКОЗ_до', 'количество_инъекций', 'интервал_дней']
    input_data[num_cols] = scaler.transform(input_data[num_cols])
    probs = model.predict_proba(input_data)[0]
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
