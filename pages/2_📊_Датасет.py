import streamlit as st
import pandas as pd

st.set_page_config(page_title="Датасет", page_icon="📊", layout="wide")

st.title("📊 Описание набора данных")
st.markdown("---")

st.markdown("""
### Предметная область
Датасет **"Predict Students' Dropout and Academic Success"** содержит информацию о студентах 
высших учебных заведений Португалии. Задача — предсказать, отчислится ли студент, 
продолжит обучение или успешно окончит программу.

### Целевая переменная
- **0 — Dropout** (Отчислен)
- **1 — Enrolled** (Продолжает обучение)
- **2 — Graduate** (Выпускник)
""")

st.markdown("---")
st.markdown("### 📋 Описание признаков")

features_info = {
    "Признак": [
        "marital_status", "application_mode", "application_order", "course",
        "daytime_evening_attendance", "previous_qualification", "previous_qualification_grade",
        "nationality", "mothers_qualification", "fathers_qualification",
        "mothers_occupation", "fathers_occupation", "admission_grade",
        "displaced", "educational_special_needs", "debtor",
        "tuition_fees_up_to_date", "gender", "scholarship_holder",
        "age_at_enrollment", "international",
        "curricular_units_1st_sem_credited", "curricular_units_1st_sem_enrolled",
        "curricular_units_1st_sem_evaluations", "curricular_units_1st_sem_approved",
        "curricular_units_1st_sem_grade", "curricular_units_1st_sem_without_evaluations",
        "curricular_units_2nd_sem_credited", "curricular_units_2nd_sem_enrolled",
        "curricular_units_2nd_sem_evaluations", "curricular_units_2nd_sem_approved",
        "curricular_units_2nd_sem_grade", "curricular_units_2nd_sem_without_evaluations",
        "unemployment_rate", "inflation_rate", "gdp",
        "total_enrolled", "total_approved", "approval_rate"
    ],
    "Описание": [
        "Семейное положение", "Способ подачи заявления", "Порядок выбора при подаче заявления", "Код курса",
        "Дневная (1) или вечерняя (0) форма", "Предыдущая квалификация", "Оценка предыдущей квалификации",
        "Национальность", "Квалификация матери", "Квалификация отца",
        "Род занятий матери", "Род занятий отца", "Оценка при поступлении",
        "Переселенец (0/1)", "Особые образовательные потребности (0/1)", "Должник (0/1)",
        "Оплата обучения актуальна (0/1)", "Пол (0=Ж, 1=М)", "Стипендиат (0/1)",
        "Возраст при поступлении", "Иностранный студент (0/1)",
        "Зачтённые предметы (1 сем.)", "Записанные предметы (1 сем.)",
        "Оценённые предметы (1 сем.)", "Сданные предметы (1 сем.)",
        "Средняя оценка (1 сем.)", "Предметы без оценки (1 сем.)",
        "Зачтённые предметы (2 сем.)", "Записанные предметы (2 сем.)",
        "Оценённые предметы (2 сем.)", "Сданные предметы (2 сем.)",
        "Средняя оценка (2 сем.)", "Предметы без оценки (2 сем.)",
        "Уровень безработицы (%)", "Уровень инфляции (%)", "ВВП",
        "Всего записанных предметов (1+2 сем.)", "Всего сданных предметов (1+2 сем.)",
        "Доля сданных от записанных"
    ]
}

st.dataframe(pd.DataFrame(features_info), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("### 🔧 Предобработка данных")
st.markdown("""
1. **Пропуски:** Отсутствуют (4424 записи, 0 null-значений)
2. **Дубликаты:** Не обнаружены
3. **Переименование столбцов:** Приведены к формату `snake_case`
4. **Feature Engineering:**
   - `total_enrolled` — суммарное количество записанных предметов за оба семестра
   - `total_approved` — суммарное количество сданных предметов за оба семестра
   - `approval_rate` — доля сданных предметов от записанных
5. **Кодирование целевой переменной:** Dropout=0, Enrolled=1, Graduate=2
""")

st.markdown("---")
st.markdown("### 📊 Обзор данных")

df = pd.read_csv('data/students_clean.csv')

tab1, tab2, tab3 = st.tabs(["Первые строки", "Статистика", "Информация"])

with tab1:
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    st.dataframe(df.describe(), use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Количество записей", df.shape[0])
        st.metric("Количество признаков", df.shape[1])
    with col2:
        st.metric("Пропущенные значения", df.isna().sum().sum())
        st.metric("Дубликаты", df.duplicated().sum())