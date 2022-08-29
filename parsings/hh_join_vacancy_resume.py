"""Объединение загруженных файлов по количеству вакансий и резюме
в итоговые файлы"""

import os
from parsing_hh import ParsingHH

hh_obj = ParsingHH()

# сбор данных по вакансиям и резюме из файлов
df_merge, df_concat = hh_obj.join_vacancy_resume(search='specialization')
# df = hh_obj.join_vacancy_resume(search='specialization', only_IT=True)
category = hh_obj.category_specializations.keys()

df_flt = df_merge.id.isin(category)
df_merge[df_flt].to_excel(os.path.join(hh_obj.path_file,
                                       'vacancy_resume.xlsx'),
                          index=False)
df_merge[df_flt].to_csv(os.path.join(hh_obj.path_file, 'vacancy_resume.csv'),
                        sep=';', index=False)

df_merge[~df_flt].to_excel(os.path.join(hh_obj.path_file,
                                        'vacancy_resume_cat.xlsx'),
                           index=False)

print(df_merge.info())
print(df_concat.info())

group_columns = df_merge.columns.values.tolist()[1:3]
df_merge = df_merge.groupby(group_columns, as_index=False).aggregate(
    {'vacancy_counts': 'mean', 'resume_counts': 'mean'})

group_columns.append('kind')
df_concat = df_concat.groupby(group_columns, as_index=False).aggregate(
    {'counts': 'mean', 'vacancy_counts': 'mean', 'resume_counts': 'mean'})

# df_merge['category'] = df_merge['id'].map(hh_obj.spec_to_category)
# df_concat['category'] = df_concat['id'].map(hh_obj.spec_to_category)

print(df_merge.info())
print(df_concat.info())

int_columns = ['counts', 'vacancy_counts', 'resume_counts']
for df, mc in zip((df_merge, df_concat), ('merge', 'concat')):
    df['category'] = df['id'].map(hh_obj.spec_to_category)
    for col in int_columns:
        if col in df.columns:
            df[col] = df[col].round().astype(int)
    df[df.id.isin(category)].to_csv(os.path.join(hh_obj.path_file,
                                                 f'vacancy_resume_{mc}_c.csv'),
                                    sep=';', encoding='cp1251', index=False)

    df[~df.id.isin(category)].to_csv(os.path.join(hh_obj.path_file,
                                                  f'vacancy_resume_{mc}.csv'),
                                     sep=';', encoding='cp1251', index=False)
