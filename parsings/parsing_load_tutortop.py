"""Парсинг сайта tutortop.ru"""

import os
import pandas as pd
from parsing_tt import ParsingTT


tt_obj = ParsingTT()

# тестовые загрузки
# url_link = 'https://tutortop.ru/courses_selection/kursy_po_tilda/'

# html_txt = tt_obj.get_courses_from_page(url=url_link, file_read=True)
# html_txt = tt_obj.get_courses_from_page(url=url_link)
# data = tt_obj.soup_partition_courses(html_txt)
# print(data.info())
# df = tt_obj.soup_partition_courses(html_txt)
# df.to_csv(os.path.join(tt_obj.path_file, 'df.csv'), sep=';', index=False)
# print(df.info())

# Формирование списка курсов по разделам в ДФ и сохранение их в файл .csv
tt_obj.make_list_courses()
# Загрузка курсов со всех страниц и сохранение ДФ в файл .csv
tt_obj.load_all_courses()
# Чтение из файла данных о загруженных курсах и очистка данных
tt_obj.preprocess_df_courses()
