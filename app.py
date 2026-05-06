import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Калькулятор анатомического ответа на анти‑VEGF", layout="wide")
st.title("🔬 Калькулятор прогноза анатомического ответа на анти‑VEGF терапию")

@st.cache_resource
def load_artifacts():
    model = joblib.load('rf_6features_анатомический.pkl')
    scaler = joblib.load('scaler_6features.pkl')
    encoder = joblib.load('label_encoder_6features.pkl')
    # Сохраним имена признаков, которые использовались при обучении (можно было сохранить отдельно, но для простоты извлечём из модели)
    feature_names = model.feature_names_in_
    return model, scaler, encoder, feature_names

model, scaler, encoder, feature_names = load_artifacts()

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
    # Создаём DataFrame с одной строкой
    input_df = pd.DataFrame({
        'возраст': [age],
        'ЦТС_до': [cts],
        'МКОЗ_до': [mkoz],
        'препарат': [drug],
        'количество_инъекций': [inj],
        'интервал_дней': [interval]
    })
    # OHE для препарата (как при обучении)
    input_df = pd.get_dummies(input_df, columns=['препарат'], drop_first=True)
    # Добавляем недостающие колонки (если какой-то категории нет)
    # Ожидаемые колонки: 'возраст', 'ЦТС_до', 'МКОЗ_до', 'количество_инъекций', 'интервал_дней',
    # 'препарат_Визкью', 'препарат_Авастин'
    expected = ['возраст', 'ЦТС_до', 'МКОЗ_до', 'количество_инъекций', 'интервал_дней',
                'препарат_Визкью', 'препарат_Авастин']
    for col in expected:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected]
    # Масштабируем числовые колонки
    num_cols = ['возраст', 'ЦТС_до', 'МКОЗ_до', 'количество_инъекций', 'интервал_дней']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    # Убеждаемся, что порядок колонок совпадает с тем, на чём обучена модель
    # (можно переиндексировать по feature_names, которые хранятся в модели)
    input_df = input_df[feature_names]
    probs = model.predict_proba(input_df)[0]
    pred_class = np.argmax(probs)
    pred_name = encoder.inverse_transform([pred_class])[0]
    classes = encoder.classes_
    prob_dict = dict(zip(classes, probs))
    st.markdown("---")
    st.subheader("📊 Результаты прогноза")
    st.markdown("**Анатомический исход (изменение отёка):**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Значительное уменьшение", f"{prob_dict.get('значительное_уменьшение', 0)*100:.1f}%")
    c2.metric("Умеренное уменьшение", f"{prob_dict.get('умеренное_уменьшение', 0)*100:.1f}%")
    c3.metric("Стабилизация", f"{prob_dict.get('стабилизация', 0)*100:.1f}%")
    c4.metric("Увеличение", f"{prob_dict.get('увеличение', 0)*100:.1f}%")
    st.markdown(f"**Наиболее вероятный исход:** {pred_name}")
    st.caption("Модель Random Forest обучена на 6 клинических параметрах. Точность прогноза – около 71%.")
