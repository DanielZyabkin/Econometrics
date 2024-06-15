import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype
import math


df = pd.read_csv('World University Rankings 2023.csv')

df = df.drop(['University Rank', 'Teaching Score', 'Research Score', 'Citations Score','International Outlook Score'], axis=1)
df = df.dropna()
df.reset_index(drop=True, inplace=True)

ranks = df.copy()


ranks = ranks.rename(columns={'Industry Income Score': 'Industry Income'})

for i in range(ranks.shape[0]):
    if ranks.at[i , 'Industry Income'] < 57.93:
        ranks.at[i , 'Industry Income'] = 0
    elif ranks.at[i , 'Industry Income'] < 78.96 and ranks.at[i, 'Industry Income'] >= 57.93:
        ranks.at[i , 'Industry Income'] = 1
    else:
        ranks.at[i , 'Industry Income'] = 2


ranks['Industry Income'].replace({0: 'низкий', 1: 'средний', 2: 'высокий'}, inplace = True)
cat_type = CategoricalDtype(categories = ['низкий', 'средний', 'высокий'], ordered = True)
ranks['Industry Income'] = ranks['Industry Income'].astype(cat_type)



ranks['No of student'] = ranks['No of student'].astype(str)
for i in range(ranks.shape[0]):
        ranks.at[i, 'No of student'] = ranks.at[i, 'No of student'].replace(',', '')


ranks['No of student'] = ranks['No of student'].astype(float)


ranks['International Student'] = ranks['International Student'].astype(str)
for i in range(ranks.shape[0]):
    if ranks.at[i, 'International Student'] == '%':
        ranks.at[i, 'International Student'] = '0%'

    
for i in range(ranks.shape[0]):
    ranks.at[i, 'International Student'] = ranks.at[i, 'International Student'].replace('%', '')
ranks['International Student'] = ranks['International Student'].astype(float)
print(ranks['International Student'].mean())


for i in range(ranks.shape[0]):
    if  ranks.at[i, 'International Student']  <= 20:
        ranks.at[i, 'International Student'] = 'низкий'
    elif ranks.at[i, 'International Student'] <= 40:
        ranks.at[i, 'International Student'] = 'средний'
    else:
        ranks.at[i, 'International Student'] = 'высокий'


        
cat_type = CategoricalDtype(categories = ['низкий', 'средний', 'высокий'], ordered = True)
ranks['International Student'] = ranks['International Student'].astype(cat_type)


# замена дефис на тире
ranks['OverAll Score'] = ranks['OverAll Score'].str.replace('–', '-')



ranks['OverAll Score'] = ranks['OverAll Score'].astype(str)
# Создаем список уникальных диапазонов
unique_ranges = ranks['OverAll Score'].unique()

# Проходимся по каждому уникальному диапазону
for range_str in unique_ranges:
    # Проверяем, является ли текущее значение числом
    if '-' not in range_str:
        # Если число, конвертируем его в числовой формат
        uniform_values = float(range_str)
        ranks.loc[ranks['OverAll Score'] == range_str, 'OverAll Score'] = uniform_values
    else:
        # Разбиваем строку диапазона на начальное и конечное значение
        start, end = map(float, range_str.split('-'))
        
        # Генерируем равномерно распределенные значения по возрастанию для текущего диапазона
        uniform_values = np.linspace(start, end, num=ranks[ranks['OverAll Score'] == range_str].shape[0])
    
    # Заменяем значения в столбце "OverAll Score" для текущего диапазона
    ranks.loc[ranks['OverAll Score'] == range_str, 'OverAll Score'] = uniform_values



ranks['OverAll Score'] = ranks['OverAll Score'].astype(float)




# преобразуем Female:Male
ranks['Female:Male Ratio'] = ranks['Female:Male Ratio'].astype(str)


def categorize_ratio(ratio):
    if ratio == 'nan':
        return np.NaN
    else:
        female, male = map(int, ratio.split(':'))
        if male == 0:
            male = female / 2
        ratio = female / male
        if ratio < 0.5:
            return 'Более мужской'
        elif 0.5 <= ratio <= 1.5:
            return 'Сбалансированный'
        else:
            return 'Более женский'
        

ranks['Female:Male Ratio'] = ranks['Female:Male Ratio'].apply(categorize_ratio)     
cat_type_female_male = CategoricalDtype(categories=['Более мужской', 'Более женский', 'Сбалансированный'], ordered=True)
ranks['Female:Male Ratio'] = ranks['Female:Male Ratio'].astype(cat_type_female_male)


# числовой анализ

CA = ranks.select_dtypes(include='float')
CA_STAT = CA.describe()
# Медиана для всех переменных
CA_med = CA.median() # Получается pandas.Series
# Межквартильный размах для всех переменных
# Вычисляется по определению
CA_iqr = CA.quantile(q=0.75) - CA.quantile(q=0.25) # Получается pandas.Series
# Создаем pandas.DataFrame из новых статистик
#W = pd.DataFrame([CA_med, CA_iqr], index=['median', 'IQR'])
W = pd.DataFrame([CA_iqr], index=['IQR'])
# Объединяем CA_STAT и W
CA_STAT = pd.concat([CA_STAT, W])

# Определение выбросов для каждого столбца
outliers = {}
for col in ['No of student', 'No of student per staff', 'OverAll Score']:
    std = np.std(CA[col])
    mean = np.mean(CA[col])
    sel_out = np.abs(CA[col] - mean) > 3 * std  # Определение выбросов
    outliers[col] = sel_out

# Создание общего фильтра для выбросов
combined_filter = np.zeros(len(CA), dtype=bool)
for col_filter in outliers.values():
    combined_filter |= col_filter

# Удаление строк с выбросами
CA_no_outliers = CA[~combined_filter]

CA_no_outliers.reset_index(drop=True, inplace=True)

# Анализ корреляции между количественными переменными
# Используем библиотеку scipy
from scipy.stats import pearsonr
from scipy.stats import spearmanr
# Здесь будут значения оценок коэффициента корреляции Пирсона
C_P = pd.DataFrame([], index=CA_no_outliers.columns, columns=CA_no_outliers.columns) 
# Здесь будут значения значимости оценок коэффициента корреляции Пирсона
P_P = pd.DataFrame([], index=CA_no_outliers.columns, columns=CA_no_outliers.columns)
# Здесь будут значения оценок коэффициента корреляции Спирмена
C_S = pd.DataFrame([], index=CA_no_outliers.columns, columns=CA_no_outliers.columns)
# Здесь будут значения значимости оценок коэффициента корреляции Спирмена
P_S = pd.DataFrame([], index=CA_no_outliers.columns, columns=CA_no_outliers.columns)

for x in CA_no_outliers.columns:
    for y in CA_no_outliers.columns:
        C_P.loc[x,y], P_P.loc[x,y] = pearsonr(CA_no_outliers[x], CA_no_outliers[y])
        C_S.loc[x,y], P_S.loc[x,y] = spearmanr(CA_no_outliers[x], CA_no_outliers[y])


# Сохраняем текстовый отчет на разные листы Excel файла
with pd.ExcelWriter('CARS_STAT.xlsx', engine="openpyxl") as wrt:
# Общая статистика
    CA_STAT.to_excel(wrt, sheet_name='stat')
# Корреляция Пирсона
    C_P.to_excel(wrt, sheet_name='Pearson')
    dr = C_P.shape[0] + 2
    P_P.to_excel(wrt, startrow=dr, sheet_name='Pearson') # Значимость
# Корреляция Спирмена
    C_S.to_excel(wrt, sheet_name='Spirmen')
    dr = C_S.shape[0] + 2
    P_S.to_excel(wrt, startrow=dr, sheet_name='Spirmen') # Значимость


unique_counts = ranks.apply(lambda x: x.nunique())
print(unique_counts)



# Анализ корреляции между количественной целевой переменной
# и качественной объясняющей
# Используем библиотеку scipy
# Критерий Крускала-Уоллиса
from scipy.stats import kruskal


# Определение выбросов для каждой количественной переменной
outliers = {}
for col in ['No of student', 'No of student per staff', 'OverAll Score']:
    std = np.std(ranks[col])
    mean = np.mean(ranks[col])
    sel_out = np.abs(ranks[col] - mean) > 3 * std  # Определение выбросов
    outliers[col] = sel_out

# Создание общего фильтра для выбросов
combined_filter = np.zeros(len(ranks), dtype=bool)
for col_filter in outliers.values():
    combined_filter |= col_filter

# Удаление строк с выбросами только из количественных переменных
ranks_no_outliers = ranks[~combined_filter]


ranks_no_outliers.reset_index(drop=True, inplace=True)


lst = ['No of student', 'No of student per staff', 'OverAll Score']
for i in lst:
    set_more_men = ranks_no_outliers['Female:Male Ratio'] == 'Более мужской'
    x_1 = ranks_no_outliers.loc[set_more_men, i]
    set_balance = ranks_no_outliers['Female:Male Ratio'] == 'Сбалансированный'
    x_2 = ranks_no_outliers.loc[set_balance, i].dropna()
    set_more_women = ranks_no_outliers['Female:Male Ratio'] == 'Более женский'
    x_3 = ranks_no_outliers.loc[set_more_women, i]

    kruskal_ = kruskal(x_1, x_2, x_3)
    with open('CARS_STAT.txt', 'a') as fln:
        print(f'kruskal-Wallis criterion for variables \'{i}\' and \'Female: Male Ratio\'',
            file=fln)
        print(kruskal_, file=fln)



    set_low = ranks_no_outliers['International Student'] == 'низкий'
    x_1 = ranks_no_outliers.loc[set_low, i]
    set_average = ranks_no_outliers['International Student'] == 'средний'
    x_2 = ranks_no_outliers.loc[set_average, i]
    set_high = ranks_no_outliers['International Student'] == 'высокий'
    x_3 = ranks_no_outliers.loc[set_more_women, i]


    kruskal_ = kruskal(x_1, x_2, x_3)


    with open('CARS_STAT.txt', 'a') as fln:
        print(f'kruskal-Wallis criterion for variables \'{i}\' and \'International student\'',
            file=fln)
        print(kruskal_, file=fln)
    
    set_low = ranks_no_outliers['Industry Income'] == 'низкий'
    x_1 = ranks_no_outliers.loc[set_low, i]
    set_average = ranks_no_outliers['Industry Income'] == 'средний'
    x_2 = ranks_no_outliers.loc[set_average, i]
    set_high = ranks_no_outliers['Industry Income'] == 'высокий'
    x_3 = ranks_no_outliers.loc[set_high, i]


    Research_score_sig = kruskal(x_1, x_2, x_3)


    with open('CARS_STAT.txt', 'a') as fln:
        print(f'kruskal-Wallis criterion for variables \'{i}\' and \'Industry Income\'',
            file=fln)
        print(Research_score_sig, file=fln)


# Анализ взаимосвязи между двумя качественными переменными
        
import statsmodels.api as sm
import itertools
import os
# Список переменных
variables = ['International Student', 'Female_Male Ratio', 'Industry Income']
folder_path = "/Users/danielzyabkin/work 2/Scripts"

# Создание всех возможных комбинаций переменных включая обратные
combinations = list(itertools.combinations(variables, 2))

ranks_no_outliers.rename(columns={'Female:Male Ratio': 'Female_Male Ratio'}, inplace=True)

# Проходим по каждой комбинации
for i, (var1, var2) in enumerate(combinations):
    # Строим таблицу сопряженности
    crtx = pd.crosstab(ranks_no_outliers[var1], ranks_no_outliers[var2], margins=True)
    crtx.columns.name = var2
    crtx.index.name = f'{var1}\{var2}'
    
    # Создаем объект sm.stats.Table для проведения анализа
    tabx = sm.stats.Table(crtx)
    
    # Создаем новый файл Excel для каждой комбинации переменных
    file_path = os.path.join(folder_path, f'Анализ_взаимосвязи_{i}.xlsx')
    with pd.ExcelWriter(file_path) as writer:
        # Таблица сопряженности
        tabx.table_orig.to_excel(writer, sheet_name=f'{var1}-{var2}'[:31])
        # Ожидаемые частоты при независимости
        tabx.fittedvalues.to_excel(writer, sheet_name=f'{var1}-{var2}'[:31], startrow=tabx.table_orig.shape[0] + 2)
        # Критерий хи-квадрат для номинальных переменных
        resx = tabx.test_nominal_association()
        # Записываем результат
        with open(f'Анализ_взаимосвязи_Хи.txt', 'a') as file:
            print(f'Chi-square criterion for variables {var1} and {var2}:\n', resx, file=file)
    

    # Рассчет Cramer's V
    nr = tabx.table_orig.shape[0]
    nc = tabx.table_orig.shape[1]
    N = tabx.table_orig.iloc[nr-1, nc-1]
    hisq = resx.statistic
    CrV = np.sqrt(hisq/(N*min((nr - 1, nc - 1))))
    # Записываем результат в файл
    with open('Анализ_взаимосвязи_Cramer.txt', 'a') as file:
        print(f'Cramer V statistics for variables {var1} and {var2}:\n', CrV, file=file)

# графический анализ качественных
        
dfn = ranks_no_outliers.select_dtypes(include=['category'])

plt.figure(figsize=(15, 20))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
nplt = 0
nrow = dfn.shape[1]



for s in dfn.columns:
    if s == 'Female_Male Ratio':
        nplt += 1
        ax = plt.subplot(nrow, 1, nplt)

        ftb = pd.crosstab(dfn[s], s)
        ftb.index.name = 'Категории'

        ftb.columns.name = None
        ftb.T.plot.bar(ax=ax, grid=True, legend=True, title=s, rot=0, table=True,
                       color={'Сбалансированный': 'yellow', 'Более мужской': 'green', 'Более женский': 'red'})
    
    else:
        nplt += 1
        ax = plt.subplot(nrow, 1, nplt)

        ftb = pd.crosstab(dfn[s], s)
        ftb.index.name = 'Категории'

        ftb.columns.name = None
        ftb.T.plot.bar(ax=ax, grid=True, legend=True, title=s, rot=0, table=True,
                       color={'средний': 'yellow', 'высокий': 'green', 'низкий': 'red'})


    ax.set_xticklabels([])

plt.show()


CA = ranks_no_outliers.copy()
crtx = pd.crosstab(CA['Female_Male Ratio'], CA['Industry Income'], margins=True)
crtx.columns.name = 'Industry Income'
crtx.index.name = 'Female_Male Ratio'
plt.figure(figsize=(15, 9))
ax = plt.subplot(2, 1, 1)
crtx.iloc[:3, :3].plot.bar(ax=ax)
#ax = plt.subplot(2, 1, 2)
#crtx.iloc[3:6, 3:6].plot.bar(ax=ax)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


crtx = pd.crosstab(CA['Female_Male Ratio'], CA['Industry Income'], margins=True)
crtx.columns.name = 'Industry Income'
crtx.index.name = 'Female_Male Ratio'
plt.figure(figsize=(15, 9))
ax = plt.subplot(2, 1, 1)
crtx.iloc[:3, :3].plot.bar(ax=ax)
#ax = plt.subplot(2, 1, 2)
#crtx.iloc[3:6, 3:6].plot.bar(ax=ax)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()  

crtx = pd.crosstab(CA['International Student'], CA['Industry Income'], margins=True)
crtx.columns.name = 'Industry Income'
crtx.index.name = 'International Student'
plt.figure(figsize=(15, 9))
ax = plt.subplot(2, 1, 1)
crtx.iloc[:3, :3].plot.bar(ax=ax)
#ax = plt.subplot(2, 1, 2)
#crtx.iloc[3:6, 3:6].plot.bar(ax=ax)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()


crtx = pd.crosstab(CA['International Student'], CA['Female_Male Ratio'], margins=True)
crtx.columns.name = 'Female_Male Ratio'
crtx.index.name = 'International Student'
plt.figure(figsize=(15, 9))
ax = plt.subplot(2, 1, 1)
crtx.iloc[:3, :3].plot.bar(ax=ax)
#ax = plt.subplot(2, 1, 2)
#crtx.iloc[3:6, 3:6].plot.bar(ax=ax)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# графический анализ количественных

dfn = ranks_no_outliers.select_dtypes(include='float64')

nrow = dfn.shape[1]
fig, ax_lst = plt.subplots(nrow, 1, figsize=(15, nrow * 10))  # Создание нескольких подграфиков
for nplt, s in enumerate(dfn.columns):  # Использование enumerate для итерации по столбцам и получения индекса
    dfn.hist(column=s, ax=ax_lst[nplt], bins='fd', legend=False, density=True)  # Построение гистограммы с относительной частотой
    ax_lst[nplt].set_title(s)  # Установка заголовка для каждого подграфика
    ax_lst[nplt].set_ylabel('Относительная частота')  # Добавление подписи к вертикальной оси
     
fig.subplots_adjust(wspace=0.5, hspace=1.0)
fig.suptitle(f'Гистограммы переменных {list(dfn.columns)}')
plt.tight_layout()  # Улучшение компактности размещения графиков
plt.subplots_adjust(top=0.95)


plt.show()


# граф. анализ связт количественной и качественной

dfn = ranks_no_outliers.copy()
cols = dfn.select_dtypes(include='category').columns

nrow = len(cols)
fig, ax_lst = plt.subplots(nrow, 1, figsize=(10, nrow * 5))  # Создание нескольких подграфиков
for nplt, s in enumerate(cols):  # Использование enumerate для итерации по столбцам и получения индекса
    dfn.boxplot(column='OverAll Score', by=s, ax=ax_lst[nplt], grid=True, notch=True, bootstrap=50,
                showmeans=True, color=None)
    ax_lst[nplt].set_ylabel('OverAll Score')  # Добавление подписи к вертикальной оси
fig.suptitle('Категоризированные диаграммы Бокса-Вискера')
plt.subplots_adjust(wspace=0.5, hspace=0.5)  # Улучшение компактности размещения графиков
plt.show()

# граф. анализ количественной и количественной

dfn = ranks_no_outliers.select_dtypes(include='float64')
nrow = dfn.shape[1] - 1
fig, ax_lst = plt.subplots(nrow, 1, figsize=(10, nrow * 5))  # Используйте figsize здесь
nplt = -1
for s in dfn.columns[:-1]:
    nplt += 1
    dfn.plot.scatter(s, 'OverAll Score', ax=ax_lst[nplt])
    ax_lst[nplt].grid(visible=True)
    ax_lst[nplt].set_title(f'Связь OverAll Score с {s}')
fig.subplots_adjust(wspace=1.0, hspace=1.0)
fig.suptitle(f'Связь OverAll Score с No of student, No of student per staff')
plt.show()


plt.figure(figsize=(10, 5))  # Размеры графика
plt.scatter(dfn['No of student'], dfn['No of student per staff'])  # Построение диаграммы рассеяния
plt.xlabel('No of student')  # Подпись по оси x
plt.ylabel('No of student per staff')  # Подпись по оси y
plt.title('Связь No of student per staff с No of student ')  # Заголовок графика
plt.grid(True)  # Отображение сетки
plt.show()  # Показать график

import scipy

plt.hist(dfn['No of student'], bins=30, density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = scipy.stats.norm.pdf(x, np.mean(dfn['No of student']), np.std(dfn['No of student']))
plt.plot(x, p, 'k', linewidth=2)

plt.ylabel('Частота')
plt.xlabel('No of student')
plt.savefig('hist_noofstudent.png')
plt.show()

plt.hist(dfn['No of student per staff'], bins=30, density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = scipy.stats.norm.pdf(x, np.mean(dfn['No of student per staff']), np.std(dfn['No of student per staff']))
plt.plot(x, p, 'k', linewidth=2)

plt.ylabel('Частота')
plt.xlabel('No of student per staff')
plt.savefig('hist_noofstudentperstaff.png')
plt.show()

plt.hist(dfn['OverAll Score'], bins=30, density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = scipy.stats.norm.pdf(x, np.mean(dfn['OverAll Score']), np.std(dfn['OverAll Score']))
plt.plot(x, p, 'k', linewidth=2)

plt.ylabel('Частота')
plt.xlabel('OverAll Score')
plt.savefig('OverAll score.png')
plt.show()

skew_data = {}
i = 0
for i in range(len(scipy.stats.skew(dfn))):
    skew_data[dfn.columns[i]] = scipy.stats.skew(dfn)[i]
skew_data



