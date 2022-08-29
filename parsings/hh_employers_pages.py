"""Загрузка работодателей по регионам (городам)"""

import os
from parsing_hh import ParsingHH

hh_obj = ParsingHH()

# каталог для сохранения загруженных страниц .json
hh_obj.json_path = os.path.join(hh_obj.path_to_save_files,
                                'employers_pages_07_22')
if not os.path.exists(hh_obj.json_path):
    os.makedirs(hh_obj.json_path)

hh_obj.get_all_employers_pages_multi(num_threads=30)

# обработка загруженных страниц
hh_obj.json_to_csv = os.path.join(hh_obj.path_to_save_files, 'employers',
                                  'pages_employers_07_22.csv')

hh_obj.read_employers_json_pages(num_threads=7)
