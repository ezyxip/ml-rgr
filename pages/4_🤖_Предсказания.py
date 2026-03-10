import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Предсказания", page_icon="🤖", layout="wide")

st.title("🤖 Получение предсказаний от моделей ML")
st.markdown("---")

# ==============================
# Загрузка моделей и вспомогательных файлов
# ==============================
@st.cache_resource
def load_models():
    models = {}
    
    # Scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Названия признаков
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # ML1: Logistic Regression
    with open('models/logistic_regression.pkl', 'rb') as f:
        models['Logistic Regression'] = pickle.load(f)
    
    # ML2: Gradient Boosting
    with open('models/gradient_boosting.pkl', 'rb') as f:
        models['Gradient Boosting'] = pickle.load(f)
    
    # ML3: CatBoost
    from catboost import CatBoostClassifier
    cb = CatBoostClassifier()
    cb.load_model('models/catboost_model.cbm')
    models['CatBoost'] = cb
    
    # ML4: Random Forest
    with open('models/random_forest.pkl', 'rb') as f:
        models['Random Forest'] = pickle.load(f)
    
    # ML5: Stacking
    with open('models/stacking_model.pkl', 'rb') as f:
        models['Stacking'] = pickle.load(f)
    
    # ML6: Neural Network
    from tensorflow.keras.models import load_model
    models['Neural Network'] = load_model('models/neural_network.keras')
    
    return models, scaler, feature_names

models, scaler, feature_names = load_models()

st.success(f"✓ Загружено моделей: {len(models)} | Признаков: {len(feature_names)}")

# ==============================
# Функция предсказания
# ==============================
TARGET_LABELS = {0: 'Dropout (Отчислен)', 1: 'Enrolled (Учится)', 2: 'Graduate (Выпускник)'}
TARGET_COLORS = {0: '🔴', 1: '🟡', 2: '🟢'}

def predict(model, model_name, X_df, scaler):
    """Получить предсказание от модели"""
    if model_name == 'CatBoost':
        X_input = X_df.values
    elif model_name == 'Neural Network':
        X_input = scaler.transform(X_df)
        pred = model.predict(X_input)
        return np.argmax(pred, axis=1)
    else:
        X_input = scaler.transform(X_df)
    
    pred = model.predict(X_input)
    return np.array(pred).flatten().astype(int)

# ==============================
# Выбор способа ввода
# ==============================
st.markdown("### Выберите способ ввода данных")
input_method = st.radio(
    "Способ ввода:",
    ["📝 Ручной ввод", "📁 Загрузка CSV файла"],
    horizontal=True
)

st.markdown("---")

# ==============================
# РУЧНОЙ ВВОД
# ==============================
if input_method == "📝 Ручной ввод":
    st.markdown("### 📝 Введите данные студента")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👤 Личные данные**")
        marital_status = st.selectbox("Семейное положение", [1, 2, 3, 4, 5, 6],
            format_func=lambda x: {1:'Холост/Не замужем', 2:'Женат/Замужем', 
                                     3:'Вдовец/Вдова', 4:'Разведён(а)', 
                                     5:'Гражданский брак', 6:'Раздельное проживание'}[x])
        gender = st.selectbox("Пол", [0, 1], format_func=lambda x: {0:'Женский', 1:'Мужской'}[x])
        age_at_enrollment = st.number_input("Возраст при поступлении (лет)", min_value=17, max_value=70, value=20)
        nationality = st.number_input("Код национальности", min_value=1, max_value=109, value=1)
        international = st.selectbox("Иностранный студент", [0, 1], format_func=lambda x: {0:'Нет', 1:'Да'}[x])
        displaced = st.selectbox("Переселенец", [0, 1], format_func=lambda x: {0:'Нет', 1:'Да'}[x])
        educational_special_needs = st.selectbox("Особые образовательные потребности", [0, 1],
            format_func=lambda x: {0:'Нет', 1:'Да'}[x])

    with col2:
        st.markdown("**🎓 Образование и поступление**")
        application_mode = st.number_input("Код способа подачи заявления", min_value=1, max_value=57, value=1)
        application_order = st.number_input("Порядок выбора (0-9)", min_value=0, max_value=9, value=1)
        course = st.number_input("Код курса", min_value=33, max_value=9991, value=9238)
        daytime_evening_attendance = st.selectbox("Форма обучения", [0, 1],
            format_func=lambda x: {0:'Вечерняя', 1:'Дневная'}[x])
        previous_qualification = st.number_input("Код предыдущей квалификации", min_value=1, max_value=43, value=1)
        previous_qualification_grade = st.number_input("Оценка предыдущей квалификации (баллы)", 
            min_value=95.0, max_value=190.0, value=130.0, step=0.1)
        admission_grade = st.number_input("Оценка при поступлении (баллы)", 
            min_value=95.0, max_value=190.0, value=127.0, step=0.1)

    with col3:
        st.markdown("**💰 Финансы и семья**")
        mothers_qualification = st.number_input("Квалификация матери (код)", min_value=1, max_value=44, value=1)
        fathers_qualification = st.number_input("Квалификация отца (код)", min_value=1, max_value=44, value=1)
        mothers_occupation = st.number_input("Занятость матери (код)", min_value=0, max_value=194, value=5)
        fathers_occupation = st.number_input("Занятость отца (код)", min_value=0, max_value=195, value=5)
        debtor = st.selectbox("Должник", [0, 1], format_func=lambda x: {0:'Нет', 1:'Да'}[x])
        tuition_fees_up_to_date = st.selectbox("Оплата обучения актуальна", [0, 1],
            format_func=lambda x: {0:'Нет', 1:'Да'}[x])
        scholarship_holder = st.selectbox("Стипендиат", [0, 1], format_func=lambda x: {0:'Нет', 1:'Да'}[x])

    st.markdown("---")
    st.markdown("**📚 Успеваемость**")
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("*1-й семестр*")
        cu_1st_credited = st.number_input("Зачтённые предметы (1 сем.)", min_value=0, max_value=26, value=0)
        cu_1st_enrolled = st.number_input("Записанные предметы (1 сем.)", min_value=0, max_value=26, value=6)
        cu_1st_evaluations = st.number_input("Оценённые предметы (1 сем.)", min_value=0, max_value=45, value=6)
        cu_1st_approved = st.number_input("Сданные предметы (1 сем.)", min_value=0, max_value=26, value=5)
        cu_1st_grade = st.number_input("Средняя оценка (1 сем.)", min_value=0.0, max_value=20.0, value=12.0, step=0.01)
        cu_1st_without = st.number_input("Предметы без оценки (1 сем.)", min_value=0, max_value=12, value=0)

    with col5:
        st.markdown("*2-й семестр*")
        cu_2nd_credited = st.number_input("Зачтённые предметы (2 сем.)", min_value=0, max_value=19, value=0)
        cu_2nd_enrolled = st.number_input("Записанные предметы (2 сем.)", min_value=0, max_value=23, value=6)
        cu_2nd_evaluations = st.number_input("Оценённые предметы (2 сем.)", min_value=0, max_value=33, value=6)
        cu_2nd_approved = st.number_input("Сданные предметы (2 сем.)", min_value=0, max_value=20, value=5)
        cu_2nd_grade = st.number_input("Средняя оценка (2 сем.)", min_value=0.0, max_value=20.0, value=12.0, step=0.01)
        cu_2nd_without = st.number_input("Предметы без оценки (2 сем.)", min_value=0, max_value=12, value=0)

    st.markdown("---")
    st.markdown("**🌍 Макроэкономические показатели**")
    col6, col7, col8 = st.columns(3)
    with col6:
        unemployment_rate = st.number_input("Уровень безработицы (%)", min_value=7.0, max_value=17.0, value=11.0, step=0.1)
    with col7:
        inflation_rate = st.number_input("Уровень инфляции (%)", min_value=-1.0, max_value=4.0, value=1.4, step=0.1)
    with col8:
        gdp = st.number_input("ВВП", min_value=-5.0, max_value=4.0, value=1.0, step=0.01)

    # Вычисление производных признаков
    total_enrolled = cu_1st_enrolled + cu_2nd_enrolled
    total_approved = cu_1st_approved + cu_2nd_approved
    approval_rate = total_approved / total_enrolled if total_enrolled > 0 else 0.0

    st.info(f"📊 Производные признаки: Всего записано: **{total_enrolled}** | "
            f"Всего сдано: **{total_approved}** | Доля сданных: **{approval_rate:.2f}**")

    # Формируем строку данных
    input_data = pd.DataFrame([[
        marital_status, application_mode, application_order, course,
        daytime_evening_attendance, previous_qualification, previous_qualification_grade,
        nationality, mothers_qualification, fathers_qualification,
        mothers_occupation, fathers_occupation, admission_grade,
        displaced, educational_special_needs, debtor,
        tuition_fees_up_to_date, gender, scholarship_holder,
        age_at_enrollment, international,
        cu_1st_credited, cu_1st_enrolled, cu_1st_evaluations,
        cu_1st_approved, cu_1st_grade, cu_1st_without,
        cu_2nd_credited, cu_2nd_enrolled, cu_2nd_evaluations,
        cu_2nd_approved, cu_2nd_grade, cu_2nd_without,
        unemployment_rate, inflation_rate, gdp,
        total_enrolled, total_approved, approval_rate
    ]], columns=feature_names)

    st.markdown("---")
    
    # Выбор модели
    selected_models = st.multiselect(
        "Выберите модели для предсказания:",
        list(models.keys()),
        default=list(models.keys())
    )

    if st.button("🔮 Получить предсказание", type="primary", use_container_width=True):
        if not selected_models:
            st.warning("Выберите хотя бы одну модель!")
        else:
            st.markdown("### 📊 Результаты предсказания")
            cols = st.columns(len(selected_models))
            
            for i, model_name in enumerate(selected_models):
                pred = predict(models[model_name], model_name, input_data, scaler)[0]
                label = TARGET_LABELS[pred]
                color = TARGET_COLORS[pred]
                
                with cols[i]:
                    st.markdown(f"**{model_name}**")
                    st.markdown(f"### {color} {label}")

# ==============================
# ЗАГРУЗКА CSV
# ==============================
else:
    st.markdown("### 📁 Загрузите CSV файл")
    st.markdown("""
    Файл должен содержать **39 столбцов** (все признаки без целевой переменной `target_encoded`).
    Если в файле отсутствуют производные признаки (`total_enrolled`, `total_approved`, `approval_rate`), 
    они будут рассчитаны автоматически.
    """)
    
    uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
    
    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)
            st.markdown(f"✓ Загружено **{upload_df.shape[0]}** записей, **{upload_df.shape[1]}** столбцов")
            
            # Удаляем целевую переменную, если она есть
            if 'target_encoded' in upload_df.columns:
                upload_df = upload_df.drop(columns=['target_encoded'])
            if 'target' in upload_df.columns:
                upload_df = upload_df.drop(columns=['target'])
            
            # Добавляем производные признаки, если их нет
            if 'total_enrolled' not in upload_df.columns:
                upload_df['total_enrolled'] = (upload_df['curricular_units_1st_sem_enrolled'] + 
                                                upload_df['curricular_units_2nd_sem_enrolled'])
            if 'total_approved' not in upload_df.columns:
                upload_df['total_approved'] = (upload_df['curricular_units_1st_sem_approved'] + 
                                                upload_df['curricular_units_2nd_sem_approved'])
            if 'approval_rate' not in upload_df.columns:
                upload_df['approval_rate'] = np.where(
                    upload_df['total_enrolled'] > 0,
                    upload_df['total_approved'] / upload_df['total_enrolled'], 0)
            
            # Проверяем наличие всех признаков
            missing_cols = set(feature_names) - set(upload_df.columns)
            if missing_cols:
                st.error(f"Отсутствуют столбцы: {missing_cols}")
            else:
                upload_df = upload_df[feature_names]
                
                with st.expander("Просмотреть загруженные данные"):
                    st.dataframe(upload_df.head(10), use_container_width=True)
                
                # Выбор модели
                selected_model = st.selectbox("Выберите модель:", list(models.keys()))
                
                if st.button("🔮 Получить предсказания", type="primary", use_container_width=True):
                    preds = predict(models[selected_model], selected_model, upload_df, scaler)
                    
                    result_df = upload_df.copy()
                    result_df['Предсказание (код)'] = preds
                    result_df['Предсказание'] = [TARGET_LABELS[p] for p in preds]
                    
                    st.markdown("### 📊 Результаты")
                    
                    # Статистика
                    col1, col2, col3 = st.columns(3)
                    dropout_count = (preds == 0).sum()
                    enrolled_count = (preds == 1).sum()
                    graduate_count = (preds == 2).sum()
                    
                    col1.metric("🔴 Dropout", dropout_count)
                    col2.metric("🟡 Enrolled", enrolled_count)
                    col3.metric("🟢 Graduate", graduate_count)
                    
                    st.dataframe(
                        result_df[['Предсказание']].join(upload_df[['age_at_enrollment', 'admission_grade', 'approval_rate']]),
                        use_container_width=True
                    )
                    
                    # Скачать результаты
                    csv_result = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Скачать результаты (CSV)",
                        csv_result,
                        "predictions.csv",
                        "text/csv",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")