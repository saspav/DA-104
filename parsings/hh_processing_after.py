import os
import re
import numpy as np
import pandas as pd
from time import time
from parsing_hh import ParsingHH

__import__('warnings').filterwarnings("ignore")

hh_obj = ParsingHH(postfix='_IT')

hh_obj.make_spec_to_category()
# print(hh_obj.spec_to_category)

specializations = hh_obj.make_dict_specializations()

# множество ИТ-специализаций
it_specs = set(hh_obj.make_dict_specializations(return_category=True,
                                                only_IT=True))
it_specs.update(hh_obj.add_it_specs)
# print(it_specs)

file_pickle = os.path.join(hh_obj.path_file, 'vacancies_prepared.pkl')
read_msg = lambda s: print(f'Читаю файл: {s}')

start_time = time()
if os.access(file_pickle, os.F_OK):
    read_msg(file_pickle)
    dfp = pd.read_pickle(file_pickle)
    print(f'Время выполнения: {time() - start_time} сек\n')
else:
    raise Exception(f'Отсутствует файл {file_pickle}')


def values_to_df(col_name, key='name'):
    return pd.DataFrame(
        pd.core.common.flatten(dfp[f'{col_name}_{key}'].dropna().tolist()),
        columns=[col_name])


# создадим ДФ из ключевых навыков
df_sk = values_to_df('key_skills')
print(f'Кол-во строк в ДФ df_sk {len(df_sk)}, '
      f'уникальных навыков {df_sk.key_skills.nunique()}')
skill_names = df_sk['key_skills'].value_counts()[:20].index
print(df_sk['key_skills'].value_counts()[:20])

# создадим ДФ из специализаций
df_sp = values_to_df('specializations')
df_sd = values_to_df('specializations', key='id')
print(f'Кол-во строк в ДФ df_sd {len(df_sd)}, '
      f'уникальных специализаций {df_sd.specializations.nunique()}')

# Подготовим ДФ для выделения ИТ специализаций
dfp['len_sp'] = dfp.specializations_id.apply(
    lambda x: 0 if pd.isna(x) else len(x))
dfp['cnt_it'] = dfp.specializations_id.apply(
    lambda x: 0 if pd.isna(x) else sum(s in it_specs for s in x))
dfp['is_it'] = dfp.apply(
    lambda row: False if pd.isna(row.specializations_id)
    else row.cnt_it >= row.len_sp / 3 or row.cnt_it > 1, axis=1)
dfp['sp_it'] = dfp.apply(lambda row: row.is_it and row.cnt_it < row.len_sp,
                         axis=1)
dfp.drop(['len_sp', 'cnt_it', 'sp_it'], axis=1, inplace=True)

# перекодировка в категории ИТ не ИТ-шных специализаций
recode_it = {'3.328': '1.327', '5.27': '1.25', '8.356': '1.110',
             '9.94': '1.89', '12.92': '1.89', '15.389': '1.225',
             '15.93': '1.89', '20.233': '1.274', '25.381': '1.82'}


def recode_to_it(row):
    text = row.specializations_id
    if row.is_it:
        if not pd.isna(text):
            if isinstance(text, (tuple, list, set)):
                result = set()
                for elem in text:
                    if elem in recode_it.keys():
                        result.add(recode_it[elem])
                    else:
                        result.add(elem)
                return tuple(sorted(result))
            elif isinstance(text, str) and text in recode_it.keys():
                return recode_it[text]
        return text
    return text


dfp['specializations_id'] = dfp.apply(lambda row: recode_to_it(row), axis=1)
# print(dfp.columns)

not_it = {128, 86, 55, 41, 115, 141, 103, 4, 85, 97, 5, 17, 21}
# чтение из файла проф.ролей не ИТ-специализаций
file_spec = os.path.join(hh_obj.path_file, 'professional_roles.xlsx')
if os.access(file_spec, os.F_OK):
    roles = pd.read_excel(file_spec)
    not_it = set(roles[roles['not_it'] == 1]['professional_roles_id'].values)

print('Проф.роли не ИТ-специализаций:',not_it)

# # старый вариант фильтрации
# temp = dfp[dfp.is_it]
# # новый вариант фильтрации
temp = dfp[~dfp['professional_roles_id'].isin(not_it)]

# преобразование значений колонок из списка в строку с разделителями '|'
for col in ['key_skills_name', 'specializations_id', 'specializations_name',
            'languages_name']:
    temp[col] = temp[col].apply(
        lambda x: '|'.join(x) if isinstance(x, (list, tuple, set)) else x)
print(temp.columns)
temp.drop('is_it', axis=1).to_csv(file_pickle.replace('.pkl', '_hh.csv'),
                                  sep=';', index=False)
# print(dfp.info())

# преобразование массива specializations_id в плоскую ячейку
print('Преобразование массива specializations_id в плоскую ячейку')
dfp = dfp.explode('specializations_id')
dfp.specializations_name = dfp.specializations_id.map(specializations)
dfp['category'] = dfp.specializations_id.map(hh_obj.spec_to_category)

# преобразование массива key_skills_name в плоскую ячейку
print('Преобразование массива key_skills_name в плоскую ячейку')
dfp = dfp.explode('key_skills_name')

# print(dfp.info())
# print(dfp.columns)


def make_id_sp(id_sp):
    sp = id_sp.split('.')
    if len(sp) < 2:
        sp.append('0')
    sp[1] = sp[1].rjust(3, '0')
    return int(''.join(sp))


# dfp.to_pickle(file_pickle)

export_columns = ['id', 'salary_min', 'salary_max',
                  'key_skills_name', 'specializations_id',
                  'specializations_name', 'is_it', 'category']
# Для ИТ
export_columns = ['id', 'key_skills_name', 'specializations_id',
                  'category']
temp = dfp[export_columns]
temp = temp[temp['specializations_id'].isin(it_specs)]
temp['id_cat'] = temp['category'].apply(lambda x: int(x.split('-')[0]))
temp['specializations_id'] = temp['specializations_id'].map(make_id_sp)
temp.dropna(subset=['key_skills_name'], inplace=True)
# сохранение файла специализаций для последующей загрузки в Yandex DataLens
temp.drop('category', axis=1).to_csv(file_pickle.replace('.pkl', '_IT.csv'),
                                     sep=';', index=False)
# список specializations_id
sp_idxs = temp['specializations_id'].unique()
print('Уникальные specializations_id:', sp_idxs)

specializations_idx = {'1.50', '1.211', '1.295', '1.270'}
temp = dfp[dfp.specializations_id.isin(specializations_idx)]
temp.key_skills_name = temp.key_skills_name.apply(
    lambda x: np.NaN if pd.isna(x) or not len(x.strip()) else x)
temp.dropna(subset=['key_skills_name'], inplace=True)
temp.to_csv(file_pickle.replace('.pkl', '_sp_null.txt'), sep=';', index=False)

# чтение файла специализации для последующей загрузки в Yandex DataLens
specs = pd.read_excel(os.path.join(hh_obj.path_file, 'specializations.xlsx'),
                      dtype=str)
specs['id_sp'] = specs['id'].map(make_id_sp)
specs['id0'] = specs['id0'].astype(int)
specs['Проф.область'] = specs.apply(
    lambda row: f"{row['id0']:02}-{row['категория']}", axis=1)
specs['Специализация'] = specs.apply(
    lambda row: f"{row['id']} {row['specializations']}", axis=1)
# specs.to_excel(os.path.join(hh_obj.path_file, 'specs.xlsx'), index=False)
specs.drop('id', axis=1, inplace=True)
specs.to_csv(os.path.join(hh_obj.path_file, 'specializations.csv'), sep=';',
             index=False)
specs = specs[specs.id_sp.isin(sp_idxs)]
specs.to_csv(os.path.join(hh_obj.path_file, 'specializations_IT.csv'), sep=';',
             index=False)
