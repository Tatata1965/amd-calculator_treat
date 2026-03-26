import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Калькулятор ВМД", layout="wide")
st.title("🔬 Калькулятор прогноза ответа на анти-VEGF терапию")


@st.cache_resource
def load_models():
    # Анатомическая (XGBoost с весами)
    anat_model = joblib.load('xgb_weighted_ЦТС_категория.pkl')
    anat_encoder = joblib.load('label_encoder_ЦТС.pkl')

    # Функциональная (улучшение зрения)
    func_model = joblib.load('rf_МКОЗ_улучшение_bin.pkl')

    # Интегрированная (успех терапии)
    int_model = joblib.load('rf_успех_терапии_bin.pkl')

    # Скейлер и данные
    scaler = joblib.load('scaler_no_leakage.pkl')
    X_mean = pd.read_csv('X_features_no_leakage.csv')

    return anat_model, anat_encoder, func_model, int_model, scaler, X_mean


anat_model, anat_encoder, func_model, int_model, scaler, X_mean = load_models()
mean_vals = X_mean.mean().to_dict()
features = X_mean.columns.tolist()

st.markdown("### Введите данные пациента")
col1, col2 = st.columns(2)

with col1:
    cts = st.number_input("ЦТС_до (мкм)", 50, 800, 300)
    mkoz = st.number_input("МКОЗ_до", 0.01, 1.0, 0.5, 0.05)
    age = st.number_input("Возраст (лет)", 50, 100, 72)

with col2:
    drug = st.selectbox("Препарат", ["Эйлеа", "Визкью", "Авастин"])
    inj = st.selectbox("Количество инъекций", [1, 2, 3])
    interval = st.slider("Интервал (дней)", 20, 60, 40)

if st.button("Прогноз"):
    # Формируем входные признаки
    inp = {f: mean_vals.get(f, 0) for f in features}
    inp['ЦТС_до'] = cts
    inp['МКОЗ_до'] = mkoz
    inp['Возраст_лет'] = age
    inp['Возраст в годах'] = age
    inp['интервал_дней'] = interval
    inp['количество_инъекций'] = inj
    drug_map = {'Эйлеа': 0, 'Визкью': 1, 'Авастин': 2}
    inp['препарат_encoded'] = drug_map[drug]

    df = pd.DataFrame([inp])
    df_scaled = pd.DataFrame(scaler.transform(df[features]), columns=features)

    # Анатомический прогноз (вероятности)
    anat_probs = anat_model.predict_proba(df_scaled)[0]  # [0,1,2,3] в порядке классов
    anat_pred_class = np.argmax(anat_probs)
    anat_pred_name = anat_encoder.inverse_transform([anat_pred_class])[0]

    # Функциональный прогноз
    func_proba = func_model.predict_proba(df_scaled)[0][1]  # вероятность улучшения

    # Интегрированный прогноз
    int_proba = int_model.predict_proba(df_scaled)[0][1]  # вероятность успеха

    st.markdown("---")
    st.subheader("📊 Результаты прогноза")

    # Анатомический (с вероятностями по каждому классу)
    st.markdown("**Анатомический исход (изменение отека):**")
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    # Классы: 0=значительное_уменьшение, 1=стабилизация, 2=увеличение, 3=умеренное_уменьшение
    with col_a1:
        st.metric("Значительное уменьшение", f"{anat_probs[0] * 100:.1f}%")
    with col_a2:
        st.metric("Умеренное уменьшение", f"{anat_probs[3] * 100:.1f}%")
    with col_a3:
        st.metric("Стабилизация", f"{anat_probs[1] * 100:.1f}%")
    with col_a4:
        st.metric("Увеличение", f"{anat_probs[2] * 100:.1f}%")

    st.markdown(f"**Наиболее вероятный исход:** {anat_pred_name}")

    st.markdown("---")
    col_f, col_i = st.columns(2)
    with col_f:
        st.metric("Вероятность улучшения зрения", f"{func_proba * 100:.1f}%")
    with col_i:
        st.metric("Вероятность успеха терапии", f"{int_proba * 100:.1f}%")

    st.caption(
        "Модели: анатомическая – XGBoost с балансировкой (точность 72%), функциональная и интегрированная – Random Forest. Обучены на 88 курсах.")