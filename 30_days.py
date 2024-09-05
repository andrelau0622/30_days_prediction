# 构建Streamlit网页
import streamlit as st
import joblib
import numpy as np
import shap
import pandas as pd

# 定义特征名称
feature_names = [
    'resp_rate_min', 'dbp_ni_max', 'spo2_min', 'urine_output', 
    'temperature_max', 'heart_rate_mean.', 'spo2_max', 
    'lactate_min', 'sbp_ni_min'
]

# 加载保存的模型
xgb_model = joblib.load('xgboost.pkl')  # 如果你使用的是.pkl文件

# Streamlit用户界面
st.title("30 Days Mortality Prediction")

# resp_rate_min: 数值输入
resp_rate_min = st.number_input("Resperate rate(min):", min_value=0.0, max_value=100.0, value=12.0)

# dbp_ni_max: 数值输入
dbp_ni_max = st.number_input("DBP max (NI):", min_value=0.0, max_value=200.0, value=80.0)

# spo2_min: 数值输入
spo2_min = st.number_input("SpO2(min):", min_value=0.0, max_value=100.0, value=95.0)

# urine_output: 数值输入
urine_output = st.number_input("Urine output:", min_value=0.0, max_value=5000.0, value=1500.0)

# temperature_max: 数值输入
temperature_max = st.number_input("Temperature(max):", min_value=30.0, max_value=45.0, value=37.0)

# heart_rate_mean.: 数值输入
heart_rate_mean = st.number_input("Heart rate(mean):", min_value=0.0, max_value=200.0, value=70.0)

# spo2_max: 数值输入
spo2_max = st.number_input("SpO2(max):", min_value=0.0, max_value=100.0, value=98.0)

# lactate_min: 数值输入
lactate_min = st.number_input("Lactate(min):", min_value=0.0, max_value=20.0, value=1.0)

# sbp_ni_min: 数值输入
sbp_ni_min = st.number_input("SBP min(NI):", min_value=0.0, max_value=200.0, value=90.0)

# 处理输入并进行预测
feature_values = [
    resp_rate_min, dbp_ni_max, spo2_min, urine_output, 
    temperature_max, heart_rate_mean, spo2_max, 
    lactate_min, sbp_ni_min
]
features = np.array([feature_values])

if st.button("Predict"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, the patient have a high risk of death."
            f"30 days mortality {probability:.1f}%。"
        )
    else:
        advice = (
            f"According to our model, the patient have a low risk of death."
            f"30 days mortality {probability:.1f}%。"
        )
    st.write(advice)

    # 计算 SHAP 值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
