import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Визуализации", page_icon="📈", layout="wide")

st.title("📈 Визуализации зависимостей в данных")
st.markdown("---")

df = pd.read_csv('data/students_clean.csv')

target_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
df['target_label'] = df['target_encoded'].map(target_map)

# ==============================
# Визуализация 1: Распределение целевой переменной
# ==============================
st.markdown("### 1. Распределение целевой переменной (статус студента)")

fig1, ax1 = plt.subplots(figsize=(8, 5))
order = ['Graduate', 'Dropout', 'Enrolled']
colors = sns.color_palette('viridis', 3)
ax = sns.countplot(x='target_label', data=df, order=order, palette='viridis', ax=ax1)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=12,
                xytext=(0, 5), textcoords='offset points')

ax1.set_title('Распределение статусов студентов')
ax1.set_xlabel('Статус')
ax1.set_ylabel('Количество студентов')
st.pyplot(fig1)

st.markdown("""
**Вывод:** Классы несбалансированы. Выпускников (Graduate) почти в 3 раза больше, 
чем студентов, продолжающих обучение (Enrolled). Это влияет на качество предсказания 
класса Enrolled.
""")

st.markdown("---")

# ==============================
# Визуализация 2: Корреляция с целевой переменной
# ==============================
st.markdown("### 2. Топ факторов, влияющих на статус студента")

fig2, ax2 = plt.subplots(figsize=(10, 8))

numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr(method='spearman')['target_encoded'].sort_values(ascending=False)
correlation = correlation.drop(['target_encoded'])

top_corr = pd.concat([correlation.head(10), correlation.tail(10)])

sns.barplot(x=top_corr.values, y=top_corr.index, palette='coolwarm', ax=ax2)
ax2.set_title('Корреляция признаков с целевой переменной (Spearman)')
ax2.set_xlabel('Коэффициент корреляции')
ax2.axvline(0, color='grey', linewidth=0.8)
st.pyplot(fig2)

st.markdown("""
**Вывод:** Наиболее сильные положительные факторы — `approval_rate`, `total_approved`, 
оценки и количество сданных предметов. Негативные факторы — возраст при поступлении, 
наличие долгов, статус должника.
""")

st.markdown("---")

# ==============================
# Визуализация 3: Влияние ключевых факторов
# ==============================
st.markdown("### 3. Влияние возраста, долгов и успеваемости на статус")

fig3, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(x='target_label', y='age_at_enrollment', data=df,
            palette='Set2', ax=axes[0], order=order)
axes[0].set_title('Возраст при поступлении')
axes[0].set_xlabel('Статус')
axes[0].set_ylabel('Возраст')

sns.countplot(x='debtor', hue='target_label', data=df,
              palette='viridis', ax=axes[1], hue_order=order)
axes[1].set_title('Влияние долгов на статус')
axes[1].set_xlabel('Должник (0=Нет, 1=Да)')
axes[1].set_ylabel('Количество')
axes[1].legend(title='Статус')

sns.boxplot(x='target_label', y='approval_rate', data=df,
            palette='coolwarm', ax=axes[2], order=order)
axes[2].set_title('Доля сданных предметов')
axes[2].set_xlabel('Статус')
axes[2].set_ylabel('Approval Rate')

plt.tight_layout()
st.pyplot(fig3)

st.markdown("""
**Выводы:**
- **Возраст:** Отчисленные студенты в среднем старше выпускников
- **Долги:** Среди должников значительно выше процент отчислений
- **Успеваемость:** У выпускников доля сданных предметов близка к 1.0, у отчисленных — к 0
""")

st.markdown("---")

# ==============================
# Визуализация 4: Heatmap корреляций ключевых признаков
# ==============================
st.markdown("### 4. Тепловая карта корреляций ключевых признаков")

key_features = [
    'age_at_enrollment', 'admission_grade', 'previous_qualification_grade',
    'curricular_units_1st_sem_approved', 'curricular_units_1st_sem_grade',
    'curricular_units_2nd_sem_approved', 'curricular_units_2nd_sem_grade',
    'approval_rate', 'total_approved', 'debtor',
    'tuition_fees_up_to_date', 'scholarship_holder', 'target_encoded'
]

fig4, ax4 = plt.subplots(figsize=(12, 10))
corr_matrix = df[key_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, ax=ax4, square=True)
ax4.set_title('Корреляционная матрица ключевых признаков')
plt.tight_layout()
st.pyplot(fig4)

st.markdown("""
**Вывод:** Видна сильная корреляция между оценками 1-го и 2-го семестров, 
а также между количеством сданных предметов и долей сданных (`approval_rate`). 
Оплата обучения и стипендия также связаны с успешным завершением.
""")