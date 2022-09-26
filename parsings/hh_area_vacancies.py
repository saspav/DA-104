"""Загрузка вакансий по регионам"""
import json

import pandas as pd
from os import path
from parsing_hh import ParsingHH

hh_obj = ParsingHH(postfix='_IT_09')

# количество поток для загрузки страниц
# hh_obj.number_url_threads = 20

# # загрузка регионов
# hh_obj.make_queue_areas_ids(areas_idxs=[8])
# print(hh_obj.areas.info())
# print(hh_obj.areas)
#
# area_id = 1
# data = hh_obj.get_employer_all_vacancies(area=area_id, specialization='1.82')
# # сохраняем в .csv если были найдены вакансии
# file_area_df_csv = hh_obj.make_file_area_df_csv(area_id)
# data.to_csv(file_area_df_csv, sep=';', index=False)

# hh_obj.load_df_vacancies_from_csv()
# print(hh_obj.file_vacancies)
# print(hh_obj.df_vacancies.info())

# загрузка вакансий по списку регионов
# hh_obj.areas_vacancies_multi(num_threads=1, areas_idxs=[2385])

loading = True
if loading:
    # весь список регионов
    areas_idx = set(hh_obj.get_areas()['area_id'].values)
    # список обработанных регионов
    areas_log = hh_obj.read_parsed_employers_log()
    # список регионов с ошибками (нет ИТ вакансий)
    areas_err = hh_obj.read_parsed_employers_log_errors()
    # множество регионов для обработки = все - загруженные - ошибочные
    areas_idxs = sorted(areas_idx - areas_log - areas_err)

    if areas_idxs:
        # установка рандомной задержки между запросами
        # hh_obj.delay = hh_obj.set_time_sleep(2.5)
        # в один поток чтобы не словить капчу
        # hh_obj.areas_vacancies_multi(num_threads=11, areas_idxs=areas_idxs)

        hh_obj.areas_vacancies_multi(areas_idxs=areas_idxs)

        # # Формирование очереди вакансий для парсинга
        # hh_obj.make_queue_areas_ids(areas_idxs=areas_idxs)

    else:
        print('Нет регионов для обработки')

# # объединение файлов с вакансиями
# hh_obj.make_df_vacancies_from_csv()

# проверим количество вакансий по всем специализациям ИТ
area_id = '113'
it_sps = ['1'] + hh_obj.add_it_specs
params = {'area': area_id, 'specialization': it_sps, 'per_page': 3}
data = hh_obj.download_page("https://api.hh.ru/vacancies", params)
found = data['found']
print(f'Регион: {area_id}, ИТ вакансий: {found}')
