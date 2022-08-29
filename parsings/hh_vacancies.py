from os import path
from parsing_hh import ParsingHH

hh_obj = ParsingHH()

# # Объединение файлов с работодателями
# hh_obj.make_all_employers()

# загрузка регионов России
areas = hh_obj.get_areas()

# просмотр загруженных работодателей
hh_obj.read_df_from_csv()
print(hh_obj.df.info())
print('Количество работодателей:', len(hh_obj.df))
print('Количество работодателей с вакансиями:',
      len(hh_obj.df[hh_obj.df.open_vacancies > 0]))

ru_hh = hh_obj.df[hh_obj.df.area_id.isin(areas.area_id)]
print('Кол-во работодателей в России:', len(ru_hh))
print('Из них с открытыми вакансиями:',
      len(ru_hh[ru_hh.open_vacancies > 0]), '\n')

# Работодатели с вакансиями не умещающимися в одном запросе
print('Работодатели с вакансиями не умещающимися в одном запросе:')
ru_hh['employer_name'] = ru_hh['employer_name'].str.slice(stop=32)
print(ru_hh[ru_hh.open_vacancies > 2000])

# Основной Этап !!!
# Многопоточная загрузка вакансий работодателей
# hh_obj.employers_vacancies_multi()

# загрузка вакансий из файла с работодателями
# hh_obj.file_csv = path.join(hh_obj.path_file, 'employers_9040000.csv')
# hh_obj.employers_vacancies_multi(num_threads=20, max_employers=5000)

# загрузка вакансий для списка работодателей их ID
# hh_obj.employers_vacancies_multi(num_threads=1, idxs_employers=[1740])

# загрузка вакансий для списка работодателей c криво загруженными данными
# - на текущий момент ошибка исправлена
# idxs_employers = hh_obj.make_bad_df_vacancies_from_csv()
# print(idxs_employers)
# hh_obj.employers_vacancies_multi(num_threads=5,
#                                  idxs_employers=idxs_employers)

# загрузка вакансий для списка работодателей c криво загруженными вакансиями
# не сразу догадался обработать капчу, чтобы не перегружать все файлы
# - дозагрузить часть отсутствующих данных по вакансиям
# idxs_employers = hh_obj.incomplete_df_vacancies_from_csv_multi(num_threads=9)
# print(f'Работодателей с неполными вакансиями: {len(idxs_employers)}')
# hh_obj.load_missing_vacancy_fields = True
# hh_obj.employers_vacancies_multi(num_threads=1,
#                                  idxs_employers=idxs_employers)
