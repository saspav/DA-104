"""Загрузка работодателей методом перебора по ID"""

from os import path
from parsing_hh import ParsingHH

hh_obj = ParsingHH()

# количество поток для загрузки страниц
hh_obj.number_url_threads = 20

# # загрузка регионов
# hh_obj.get_areas()
# print(hh_obj.areas.info())
# print(hh_obj.areas)
#
# просмотр загруженных работодателей
hh_obj.read_df_from_csv()
print(hh_obj.df.info())
print(hh_obj.df[hh_obj.df.open_vacancies > 2000])
print('Количество работодателей с вакансиями:',
      len(hh_obj.df[hh_obj.df.open_vacancies > 0]))

hh_obj.file_csv = path.join(hh_obj.path_file, 'employers_9040000.csv')
# all_employers = hh_obj.get_employers_multi(start_id=9000000)
all_employers = hh_obj.get_employers_multi()
# print(all_employers.tail())
