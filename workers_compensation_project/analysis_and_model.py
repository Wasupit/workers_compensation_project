import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def preprocess_data(df):
    data = df.copy()
    data['DateTimeOfAccident'] = pd.to_datetime(data['DateTimeOfAccident'])
    data['DateReported'] = pd.to_datetime(data['DateReported'])
    data['AccidentMonth'] = data['DateTimeOfAccident'].dt.month
    data['AccidentDayOfWeek'] = data['DateTimeOfAccident'].dt.dayofweek
    data['ReportingDelay'] = (data['DateReported'] - data['DateTimeOfAccident']).dt.days

    data.drop(columns=['DateTimeOfAccident', 'DateReported'], inplace=True)
    categorical_columns = ['Gender', 'MaritalStatus', 'PartTimeFullTime', 'ClaimDescription']
    label_encoders = {}

    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    numerical_features = ['Age', 'DependentChildren', 'DependentsOther',
                          'WeeklyPay', 'HoursWorkedPerWeek',
                          'DaysWorkedPerWeek', 'InitialCaseEstimate',
                          'AccidentMonth', 'AccidentDayOfWeek', 'ReportingDelay']

    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data, scaler, label_encoders


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return mae, rmse, r2, y_pred


def analysis_and_model_page():
    st.title("Прогнозирование стоимости страховых выплат")

    if st.button("Загрузить данные"):
        with st.spinner("Загрузка данных"):
            data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
            df = data.frame
            st.session_state['df'] = df
            st.success("Данные загружены!")

    if 'df' in st.session_state:

        df = st.session_state['df']
        st.write(df.head())

        data, scaler, label_encoders = preprocess_data(df)

        X = data.drop(columns=['UltimateIncurredClaimCost'])
        y = data['UltimateIncurredClaimCost']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        model_choice = st.selectbox("Выберите модель",
                                    ["Linear Regression", "Random Forest", "XGBoost", "Ridge"])

        if model_choice == "Linear Regression":
            model = LinearRegression()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_choice == "XGBoost":
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        else:
            model = Ridge(alpha=1.0)

        model.fit(X_train, y_train)

        mae, rmse, r2, y_pred = evaluate_model(model, X_test, y_test)

        st.subheader("Метрики модели")
        st.write(f"MAE: {mae:.2f}")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R2: {r2:.4f}")

        fig = plt.figure()
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.xlabel("Реальные значения")
        plt.ylabel("Предсказания")
        st.pyplot(fig)

        if hasattr(model, "feature_importances_"):
            st.subheader("Важность признаков")
            importances = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            st.write(importances.head(10))

        st.header("Предсказание нового случая")

        with st.form("prediction_form"):
            age = st.number_input("Возраст", 13, 76, 35)
            weekly_pay = st.number_input("Weekly Pay", 0, 5000, 500)
            initial_estimate = st.number_input("Initial Estimate", 0, 100000, 5000)

            submitted = st.form_submit_button("Предсказать")

            if submitted:
                st.write("Функционал демонстрационный — требуется расширить ввод всех признаков.")