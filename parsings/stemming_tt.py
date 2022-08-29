"""Поиск курсов для шаблона вакансий, специализаций и ключевых навыков"""

import os
import re

import numpy as np
import pandas as pd
import pymorphy2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from fuzzywuzzy import fuzz
from parsing_tt import ParsingTT

snowball = SnowballStemmer(language="russian")
morph = pymorphy2.MorphAnalyzer()
ru_stop = stopwords.words('russian') + ['nan']
pattern_punct = '[-!@"“’«»$%&\'(),/:;<=>?^_`{|}~\[\]]'
digits = [str(i) for i in range(10)]

df_all = pd.DataFrame()


def preprocess(tokens):
    """
    В функции отфильтруем также числа, проверив первый символ.
    Так, это поможет избавиться от «2019», но не от Covid-19.
    :param tokens: список слов
    :return: список нормализованных слов
    """
    return [morph.parse(word)[0].normal_form for word in tokens if
            (word[0] not in digits and word not in ru_stop)]


def normalize_text(input_text):
    """
    Перевод текста в нижний регистр, удаление дат, удаление знаков пунктуации
    деление текста на слова
    :param input_text: текст
    :return: список слов
    """
    # убираем из текста знаки пунктуации
    input_text = re.sub(pattern_punct, ' ', input_text.lower())
    input_text = input_text.replace('c++', 'cплюс').replace('1c', '1с')
    # words = word_tokenize(input_text, language='russian')
    words = input_text.split()
    out_words = []
    for word in words:
        if len(word) > 1 and word not in out_words:
            out_words.append(word)
    return sorted(out_words)


def stem_text(list_words):
    stem_words = []
    for word in list_words:
        stem_word = snowball.stem(word)
        if stem_word not in ru_stop and stem_word not in stem_words:
            stem_words.append(stem_word)
    return ' '.join(stem_words)


def lemm_text(list_words):
    lemm_words = []
    for word in list_words:
        lemm_word = morph.parse(word)[0].normal_form
        if lemm_word not in ru_stop and lemm_word not in lemm_words:
            lemm_words.append(lemm_word)
    return ' '.join(lemm_words)


def normalize_str(input_text):
    words = normalize_text(input_text)
    return stem_text(words), lemm_text(words)


def calc_ratio(pattern, text):
    len_pattern = len(pattern.split())
    len_text = len(text.split())
    ratio = fuzz.token_set_ratio(pattern, text)
    if len_text == 1 and len_pattern > len_text and ratio > 99:
        ratio /= 2
    return round(ratio, 1)


def find_courses(search_pattern):
    stem_patt, lemm_patt = normalize_str(search_pattern)
    print(f"Паттерны. Стем:{locals()['stem_patt']} "
          f"Лемма:{locals()['lemm_patt']}")

    for name_col in process_columns:
        ratio_cols = []
        for _col in ('stem', 'lemm'):
            pattern = locals()[f'{_col}_patt']
            new_col = f'{_col}_{name_col}'
            ratio_col = f'{new_col}_ratio'
            ratio_cols.append(ratio_col)
            df[ratio_col] = df[new_col].apply(lambda x: calc_ratio(pattern, x))

        df[f'{name_col}_ratio'] = df[ratio_cols].values.mean(1)

    df['all_ratio'] = df[[col for col in df.columns.values
                          if col.endswith('ratio')]].values.mean(1).round(1)

    # print(df.columns)

    ratio_cols = ['partition_ratio', 'course_title_ratio']
    limit_percent = 95
    df['found'] = df.apply(
        lambda row: any(row[col] >= limit_percent for col in ratio_cols),
        axis=1)
    df.sort_values(ratio_cols, ascending=False, inplace=True)
    # df.to_excel(tt_obj.processed_courses.replace('.csv', '_finds.xlsx'),
    #             index=False)
    # print(df.columns)

    col_name = ['course_id', 'category', 'partition', 'course_title',
                'date_begin', 'duration', 'price', 'rassrochka', 'rating',
                'school', 'school_reviews', 'discount', 'course_url',
                'top_school', 'all_ratio']
    df_fnd = df[df.found][col_name]
    df_fnd.school = df_fnd.school.str.replace('&amp;', '&')
    for quantile in (0.05, 0.1, 0.15, 0.2):
        Q1 = df_fnd.price.quantile(quantile)
        Q3 = df_fnd.price.quantile(1 - quantile)
        str_q = str(quantile).split('.')[-1]
        df_fnd[f'IQR_{quantile}'] = df_fnd.price.apply(
            lambda x: int(Q1 <= x <= Q3))
    df_fnd.sort_values('all_ratio', ascending=False, inplace=True)
    df_fnd.drop_duplicates('course_id', inplace=True)

    return df_fnd


def round_to_int(x_value, x_step=100):
    if pd.isna(x_value):
        return x_value
    if x_step > 9:
        return int((x_value + x_step / 2) // x_step * x_step)
    return round(x_value, x_step)


def save_df_to_files(data, kind):
    # сохраняем найденные курсы в файлы
    data.to_pickle(tt_obj.processed_courses.replace('.csv', f'_{kind}.pkl'))
    data.to_csv(tt_obj.processed_courses.replace('.csv', f'_{kind}.csv'),
                index=False, sep=';')
    data.to_excel(tt_obj.processed_courses.replace('.csv', f'_{kind}.xlsx'),
                  index=False)


tt_obj = ParsingTT()

process_columns = ['partition', 'course_title']

process_file = False

if process_file:
    # Чтение из файла данных о загруженных курсах
    tt_obj.read_df_processed_courses()
    df = tt_obj.df_courses_proc.copy(deep=True)
    # print(df.columns)
    # print(df.info())

    for name_col in process_columns:
        df[f'norm_{name_col}'] = df[name_col].apply(normalize_text)
        for _col in ('stem', 'lemm'):
            df[f'{_col}_{name_col}'] = df[f'norm_{name_col}'].apply(
                globals()[f'{_col}_text'])

    # print(df.info())
    save_df_to_files(df, 'stem')

df = pd.read_pickle(tt_obj.processed_courses.replace('.csv', '_stem.pkl'))
# print(df.info())

file_xls = os.path.join(tt_obj.path_file,
                        'ТОП-20 вакансии, специализации, навыки.xlsx')
# xls_top = pd.read_excel(file_xls, nrows=1)
xls_top = pd.read_excel(file_xls)
xls_top.columns = ['vacancy', 'vacancy_counts',
                   'specialization', 'specialization_counts',
                   'key_skill', 'key_skill_counts']
process_file_xls_columns = xls_top.columns.to_list()[::2]
# print(process_file_xls_columns)

measures_name = [f'{top}_{col}_{measure}' for col in ('duration', 'price')
                 for top in ('all', 'top') for measure in ('median', 'mean')]
# print(measures)

df_top = pd.DataFrame(columns=['name_type', 'position', 'name', 'counts',
                               'course_id', *measures_name])
recode_name = {'vacancy': 'Вакансия',
               'specialization': 'Специализация',
               'key_skill': 'Ключевой навык'}

for idx, name_col in enumerate(process_file_xls_columns):
    for row in xls_top.itertuples(index=True):
        print(f'Тип: {name_col} {recode_name[name_col]}')
        search_patt = row[idx * 2 + 1]
        df_found = find_courses(search_patt)
        if not len(df_found):
            df_found.loc[0] = [np.NaN] * len(df_found.columns)
            # df_found.course_id.fillna(0, inplace=True)
            # df_found.course_title.fillna('', inplace=True)
        search_pattern = search_patt[:1].upper() + search_patt[1:]
        df_found.insert(0, 'counts', row[idx * 2 + 2])
        df_found.insert(0, 'name', search_pattern)
        df_found.insert(0, 'position', row.Index + 1)
        df_found.insert(0, 'name_type', recode_name[name_col])
        course_idxs = set(df_found['course_id'].values)
        # df_found = df[df.course_id.isin(course_idxs)]
        if len(df_found):
            measures = []
            for col, rnd in dict(duration=1, price=100).items():
                flt1 = df_found[col] > 0
                for flt2 in (True, df_found['top_school']):
                    tmp = df_found[flt1 & flt2][col]
                    # округлим медиану и среднее до числа из словаря
                    measures.append(round_to_int(tmp.median(), rnd))
                    measures.append(round_to_int(tmp.mean(), rnd))
        else:
            course_idxs = []
            measures = [np.NaN] * len(measures_name)

        df_top.loc[len(df_top)] = [recode_name[name_col],
                                   row.Index + 1,
                                   search_pattern,
                                   row[idx * 2 + 2],
                                   sorted(course_idxs),
                                   *measures]
        if len(df_all):
            df_all = pd.concat([df_all, df_found], axis=0, ignore_index=True)
        else:
            df_all = df_found

# прочитаем заново инфу о курсах
tt_obj.read_df_processed_courses()
# ДФ + инфа про курсы
# df_all = df_top[df_top.columns[:5]]
# df_all = df_all.explode('course_id')
# df_all = df_all.merge(tt_obj.df_courses_proc, on='course_id', how='left')
# сохраняем найденные курсы в файлы
save_df_to_files(df_all, 'search_full')
# идентификаторы курсов соединим в строку
df_top.course_id = df_top.course_id.apply(lambda x: ','.join(map(str, x)))
# сохраняем найденные курсы в файлы
save_df_to_files(df_top, 'search')

# print(df_top.info())
# print(df_all.columns)

# search_patt = '1C разработка'
# print(find_courses('', 0,  search_patt))
