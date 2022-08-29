""" Загрузка отсутствующих вакансий по работодателям, после этапа
загрузки вакансий по работодателям hh_vacancies.py и словили капчу"""

import time
import random
from parsing_hh import ParsingHH


def set_time_sleep(time_sleep):
    time_sleep = round(random.random() * time_sleep, 1)
    print(f'time_sleep={time_sleep}')
    return time_sleep


hh_obj = ParsingHH()

# hh_obj.delay = 1.5
# hh_obj.incomplete_vacancies_multi(num_threads=1, max_rows=2000)

captcha_count = 0
for _ in range(1000):
    hh_obj.delay = set_time_sleep(2.5)
    hh_obj.new_vacancy_rows = 0
    hh_obj.errors_captcha_required = False
    hh_obj.incomplete_vacancies_multi(num_threads=1,
                                      # max_rows=random.randint(69, 99)
                                      )
    if not hh_obj.new_vacancy_rows and hh_obj.errors_captcha_required:
        captcha_count += 1
    if captcha_count > 9:
        print('Нужен перерыв. Прерываю выполнение программы!!!')
        break
    time.sleep(set_time_sleep(9))
    if hh_obj.errors_captcha_required:
        time.sleep(set_time_sleep(33))
