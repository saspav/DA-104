"""Подготовка данных для дашборда Рейтинг отношения вакансий и курсов"""

import os
import numpy as np
import pandas as pd
from parsing_hh import ParsingHH

__import__('warnings').filterwarnings("ignore")

hh_obj = ParsingHH()

file_csv = os.path.join(hh_obj.path_file, 'processed_tutortop_search_full.csv')
df = pd.read_csv(file_csv, sep=';', usecols=range(5))
print(df.info())
print(df.columns)

cnt = df.groupby(['name_type', 'position', 'name', 'counts'],
                 as_index=False).course_id.count().rename(
    columns={'course_id': 'courses'})

# print(cnt)

grp = cnt.groupby('name_type', as_index=False).aggregate({'counts': sum,
                                                          'courses': sum})
grp.columns = ['name_type', 'total_search', 'total_courses']

# print(grp)

cnt = cnt.merge(grp, on='name_type', how='left')
cnt['Доля в поиске'] = cnt.counts / cnt.total_search * 100
cnt['Доля курсов'] = cnt.courses / cnt.total_courses * 100
cnt['Рейтинг'] = cnt['Доля в поиске'] / cnt['Доля курсов']
# cnt['Рейтинг'] = cnt['Рейтинг'] / (1 + cnt['position'] / 10)
cnt['Рейтинг'] = cnt['Рейтинг'] / cnt['position'] * 10
cnt['Рейтинг'] = cnt['Рейтинг'].apply(lambda x: 0 if x == np.inf else x)

# # доля в поиске больше 5%
# cnt = cnt[(cnt['Доля в поиске'] > 5) & (cnt.courses > 0)]

cnt.sort_values('Рейтинг', ascending=False, inplace=True)

for col in ('Доля в поиске', 'Доля курсов', 'Рейтинг'):
    cnt[col] = cnt[col].round(1)

print(cnt.columns)

cnt.columns = ['Тип поиска', 'Позиция', 'Шаблон поиска', 'Количество Вакансий',
               'Количество Курсов', 'Всего Вакансий', 'Всего Курсов',
               'Доля вакансий', 'Доля курсов',
               'Отношение долей: вакансий/курсов']

cnt.to_csv(file_csv.replace('.csv', '_with_rating.csv'), index=False, sep=';')
cnt.to_excel(file_csv.replace('.csv', '_with_rating.xlsx'), index=False)
