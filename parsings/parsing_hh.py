import os
import re
import time
import numpy as np
import pandas as pd
import json
import queue
import random
import threading
import requests
from requests.exceptions import HTTPError, ConnectionError, Timeout
from datetime import datetime, timedelta

from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
from ast import literal_eval
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm

from stepik_addons import MakeHTML

__import__('warnings').filterwarnings("ignore")


def print_time(time_start):
    """
    Печать времени выполнения процесса
    :param time_start: время запуска в формате time.time()
    :return:
    """
    time_apply = time.time() - time_start
    hrs = time_apply // 3600
    mns = time_apply % 3600
    sec = mns % 60
    print(f'Время обработки: {hrs:.0f} час {mns // 60:.0f} мин {sec:.1f} сек')


def is_null(item):
    if pd.isna(item):
        return True
    item = item.strip()
    if not item or item in ('[]',):
        return True
    return False


def marking_nan(item):
    if is_null(item):
        return np.NaN
    return item


STAPLES = {'{': '}', '[': ']'}


def get_item_from_dict(cell_text, key_name='name'):
    """
    Функция получения данных из текста ячейки
    :param cell_text: текст
    :param key_name: ключ по которому нужно получить значение
    :return: значение или список значений, если входная строка "список"
    """
    value = np.NaN
    if not pd.isna(cell_text):
        cell_text = cell_text.strip()
        ch_fst = cell_text[0]
        ch_lst = cell_text[-1]
        # открывающая и закрывающая скобки должны быть и быть одинаковыми
        if ch_fst in STAPLES.keys() and ch_lst == STAPLES[ch_fst]:
            result = literal_eval(cell_text)
            if isinstance(result, dict):
                value = result.get(key_name, np.NaN)
            elif isinstance(result, list):
                values = []
                for item in result:
                    value = item.get(key_name, np.NaN)
                    if not pd.isna(value):
                        values.append(value)
                if values:
                    return sorted(values)
    return value


class ParsingHH:
    """
    Класс для парсинга https://api.hh.ru
    """

    def __init__(self, path_file=None):
        """
        Инициализация экземпляра класса
        :param path_file: рабочий каталог для файлов
        """
        if path_file is not None:
            self.path_file = path_file
        else:
            # self.path_file = r'D:\python-txt\DA-104_full\проект_стажировкa'
            self.path_file = r'D:\проект_стажировкa'

        self.path_to_save_files = r'D:\проект_стажировкa'
        soft_names = [SoftwareName.CHROME.value]
        op_systems = [OperatingSystem.WINDOWS.value,
                      OperatingSystem.LINUX.value]
        self.user_agent = UserAgent(software_names=soft_names,
                                    operating_systems=op_systems, limit=99)
        self.timeout = (2, 2)
        self.requests_interval = 1
        self.max_requests_number = 99
        self.areas = pd.DataFrame(columns=['area_id', 'area_name'])
        self.df = pd.DataFrame(columns=['area_id',
                                        'employer_id',
                                        'employer_name',
                                        'open_vacancies'])
        # итоговый файл с работодателями
        name_file = 'all_employers.csv'
        self.file_csv = os.path.join(self.path_file, name_file)
        # файл с логами обработки индексов работодателей
        self.file_log = os.path.join(self.path_file,
                                     name_file.replace('.csv', '.log'))
        self.idx = 1
        self.rows_in_df = 0
        self.employer_ids = set()
        self.employer_ids_excel = set()

        self.rq_time = None
        # ДФ с работодателями из постраничного парсинга
        self.df_json = pd.DataFrame(columns=self.df.columns)
        # ДФ с работодателями из экселевских файлов, которые предоставили
        # для выполнения заданий по стажировке
        self.df_excel = pd.DataFrame(columns=self.df.columns)

        # проверять ранее загруженные файлы с вакансиями
        self.checking_existing_vacancies = False
        # загружать отсутствующие поля в вакансиях
        self.load_missing_vacancy_fields = False
        # Ключи словаря для извлечения параметров вакансий
        self.dict_keys = ('id',
                          'name',
                          'salary',
                          'employment',
                          'schedule',
                          'experience',
                          'key_skills',
                          'specializations',
                          'professional_roles',
                          'languages',
                          'description',
                          'area',
                          'employer',
                          'address',
                          'created_at',
                          'published_at',
                          )

        # !!!!! включаем управление событиями !!!!!
        self.event_reader = threading.Event()
        # очередь с названием файлов, идентификаторов и т.п.
        self.files_queue = queue.Queue()
        # очередь с обработанными данными
        self.data_queue = queue.Queue()
        # каталог файлов с работодателями, собранные разными методами
        self.path_employers = os.path.join(self.path_file, 'employers')
        # каталог для сохранения загруженных страниц .json
        self.json_path = os.path.join(self.path_to_save_files,
                                      'employers_pages')
        if not os.path.exists(self.json_path):
            os.makedirs(self.json_path)
        # файл куда будем собирать работодателей со страниц .json
        self.json_to_csv = os.path.join(self.path_file, 'pages_employers.csv')
        self.pattern = re.compile("<.*?>")
        # Датафрейм с собранными вакансиями
        self.df_vacancies = pd.DataFrame()
        self.rows_in_vacancies = 0
        self.errors_captcha_required = False
        # файл с логами обработки не существующих вакансий
        self.errors_vacancies_log = os.path.join(self.path_file,
                                                 'errors_vacancies.log')
        self.errors_vacancies = set()
        # каталог для сохранения вакансий работодателя
        self.parsed_employers = os.path.join(self.path_to_save_files,
                                             'parsed_employers')
        if not os.path.exists(self.parsed_employers):
            os.makedirs(self.parsed_employers)
        # итоговый файл с вакансиями
        self.file_vacancies = os.path.join(self.path_file, 'all_vacancies.csv')
        # файл с логами обработки индексов работодателей с вакансиями
        self.parsed_employers_log = os.path.join(self.path_file,
                                                 'parsed_employers.log')
        # файл для дозагрузки неполных вакансий
        self.incomplete_vacancies = os.path.join(self.parsed_employers,
                                                 'incomplete_vacancies.csv')
        self.new_vacancy_rows = 0
        # Датафрейм с собранными неполными вакансиями
        self.df_incomplete = pd.DataFrame()
        self.rows_in_incomplete = 0
        # словарь паттернов наименований вакансий для мониторинга
        self.patterns_dict = dict()
        # словарь специализаций
        self.specializations_dict = dict()
        # словарь специализаций по категориям
        self.category_specializations = dict()
        # словарь профессиональных ролей
        self.professional_roles_dict = dict()
        # словарь профессиональных ролей по категориям
        self.category_prof_roles = dict()
        # функции для получения шаблонов поиска
        self.funcs = {'text': self.make_patterns,
                      'specialization': self.make_dict_specializations,
                      'professional_role': self.make_dict_professional_roles}
        self.filter_it = set()
        self.spec_to_category = dict()

        # время задержки в сек между запросами
        self.delay = None
        # из данных в запросе вакансий по работодателю
        # можно получить такие поля:
        self.keys_from_item = ['id', 'name', 'salary', 'schedule', 'area',
                               'employer', 'address', 'created_at',
                               'published_at']
        # эти поля можно получить только из самой вакансии
        self.keys_not_item = ['employment', 'experience', 'key_skills',
                              'specializations', 'professional_roles',
                              'languages', 'description']
        # множество индексов работодаталей для обработки
        self.idxs_employers = set()
        self.horizontal_bar = None

    def get_time_sleep(self):
        time_sleep = self.delay if self.delay else 0
        return round(random.random() * time_sleep, 2)

    def make_all_employers(self):
        """
        Процедура объединения файлов с работодателями
        :return:
        """
        # Чтение данных из основного ДФ
        self.read_df_from_csv()
        # Добавление к ним данных их других файлов с работодателями
        for file in os.listdir(self.path_employers):
            file_csv = os.path.join(self.path_employers, file)
            if file.endswith('.csv'):
                self.print_read_msg(file_csv)
                temp = pd.read_csv(file_csv, sep=';')
                self.df = pd.concat([self.df, temp], ignore_index=True)
        # заполним пустой строкой пропущенные наименования работодателя
        self.df.employer_name.fillna('', inplace=True)
        # группировкой избавимся от дублей и получим max кол-во вакансий
        self.df = self.df.groupby(['area_id', 'employer_id', 'employer_name'],
                                  as_index=False)['open_vacancies'].max()
        self.save_df_to_csv()

    def multithreaded_processing(self, num_threads, queueing, reader, writer):
        """
        Загрузка данных в несколько потоков
        :param num_threads: количество потоков
        :param queueing: формирование очереди - кортеж (имя функции, параметры)
        :param reader: читатель данных
        :param writer: писатель данных
        :return: None
        """

        # !!!!! включаем управление событиями !!!!!
        self.event_reader = threading.Event()
        self.rq_time = time.time()

        prepare_queue, *args_queue = queueing
        prepare_queue(*args_queue)

        if self.files_queue.empty():
            print('НЕТ страниц для обработки.')
        else:
            # пишем в 1 поток. Если данные писать в несколько потоков,
            # то нужно еще использовать блокировщик threading.Lock()
            twr = threading.Thread(target=writer)
            twr.start()

            # читаем и обрабатываем в Х потоков
            threads = []
            for idx_thread in range(num_threads):
                thread = threading.Thread(target=reader,
                                          args=(idx_thread + 1,))
                threads.append(thread)
                thread.start()

            # ждем, когда все потоки обработаются
            [thread.join() for thread in threads]

            # как все потоки reader завершены
            # !!!!! скажем об этом writer !!!!!
            self.event_reader.set()

    def get_employers_multi(self, num_threads=20, start_id=None,
                            numbers_id=500000, use_idx_from_excel=False):
        """
        Загрузка работодателей в несколько потоков
        :param num_threads: Количество потоков
        :param start_id: начальный индекс работодателя для парсинга hh.ru
        :param numbers_id: количество индексов для обработки
        :param use_idx_from_excel: использовать индексы из make_id_from_excel
        :return: датафрейм
        """
        # Вызов функции управления потоками и передача в неё параметров
        self.multithreaded_processing(num_threads,
                                      (self.make_queue_employer_ids,
                                       start_id,
                                       numbers_id,
                                       use_idx_from_excel),
                                      self.url_reader,
                                      self.url_writer)
        return self.df

    def remove_tags(self, html_text):
        """
        Удаление HTML тегов из текста
        :param html_text: текст HTML
        :return: очищенная от тегов строка
        """
        return re.sub(self.pattern, "", html_text)

    def make_id_from_excel(self):
        """
        Получение идентификаторов работодателей из экселевских файлов,
        которые предоставили для выполнения заданий по стажировке.
        Идентификаторы сохраняются в файл id_from_excel.txt
        :return: None
        """
        file_names_xls = ('result roles (16.06.2022).xlsx',
                          'region roles (26.05.22).xlsx')
        df = pd.DataFrame(columns=['employer_id'])
        for name_xls in file_names_xls:
            file_xls = os.path.join(self.path_file, name_xls)
            rq_time = time.time()
            self.print_read_msg(file_xls)
            # читаем из файла только колонку с работодателем
            tmp = pd.read_excel(file_xls, usecols=['employer'])
            # достаем из колонки id работодателя
            tmp['employer_id'] = tmp['employer'].apply(
                lambda x: get_item_from_dict(x, key_name='id'))
            df = pd.concat([df, pd.DataFrame(tmp['employer_id'].values,
                                             columns=['employer_id'])])
            print_time(rq_time)

        df.dropna(inplace=True)
        df['employer_id'] = df['employer_id'].astype(int)
        df.drop_duplicates(inplace=True, ignore_index=True)
        df.to_csv(os.path.join(self.path_file, 'id_from_excel.txt'),
                  index=False)

    def read_employers_id_from_excel(self):
        """
        Получение идентификаторов работодателей из файла id_from_excel.txt и
        запись их в множество self.employer_ids_excel
        :return: None
        """
        file_xls = os.path.join(self.path_file, 'id_from_excel.txt')
        if os.access(file_xls, os.F_OK):
            self.print_read_msg(file_xls)
            df = pd.read_csv(file_xls)
            self.employer_ids_excel = set(df.employer_id.values)

    def read_employers_from_excel(self):
        """
        Получение работодателей из файла employers_from_excel.csv и
        запись их идентификаторов в множество self.employer_ids_excel
        :return: None
        """
        file_xls = os.path.join(self.path_file, 'employers_from_excel.csv')
        if os.access(file_xls, os.F_OK):
            self.print_read_msg(file_xls)
            self.df_excel = pd.read_csv(file_xls, sep=';')
            self.employer_ids_excel = set(self.df_excel.employer_id.values)

    @staticmethod
    def print_read_msg(file_name):
        print(f'Читаю файл: {file_name}')

    def download_page(self, url, params=dict(), ret_json=True):
        """
        Загрузка страницы
        :param url: ссылка
        :param params: параметры
        :param ret_json: ответ в формате .json (ret_json=True)
        :return: ответ
        """
        # Значение таймаута будет применяться как к таймаутам подключения,
        # так и к таймаутам чтения. Для того, что бы установить значения
        # таймаутов отдельно, необходимо указать кортеж:
        for _ in range(self.max_requests_number):
            headers = {'User-Agent': self.user_agent.get_random_user_agent()}
            try:
                request = requests.get(url, params=params, headers=headers,
                                       timeout=self.timeout)
                request.raise_for_status()
            except ConnectionError as connection_error:
                print(f"Connection error: {connection_error}")
            except Timeout as time_out:
                print(f"Timeout error: {time_out}")
            except HTTPError as http_error:
                # print(f"HTTP error: {http_error}")
                # print(f"Нет страницы: {url}")
                if ret_json and request.json().get('errors', None) is not None:
                    if request.json().get('description', None) is None:
                        # словили ошибку про требование ввода капчи --> ставим
                        # глобальный флаг, чтобы больше не читать вакансии
                        self.errors_captcha_required = True
                return request.json() if ret_json else request
            else:
                return request.json() if ret_json else request

            print(f"Повтор запроса {url} через {self.requests_interval} сек")
            time.sleep(self.requests_interval)

        raise HTTPError(f"Страница {url} не загрузилась")

    def read_json(self, file_json):
        """
        Чтение .json файла
        :param file_json: полнное имя файла
        :return: объект .json
        """
        self.print_read_msg(file_json)
        with open(file_json, encoding='utf-8') as file:
            return json.load(file)

    def read_idx_log(self):
        """
        Чтение из лог-файла идентификаторов, которых нет на сайте hh.ru
        :return: множество идентификаторов, со значением больше, чем self.idx
        - это индекс максимального id из основного файла работодателей
        """
        if os.access(self.file_log, os.F_OK):
            tmp = pd.read_csv(self.file_log, header=None)
            tmp.columns = ['idx']
            print(f'Идентификаторы. Min: {tmp.idx.min()}, '
                  f'Max: {tmp.idx.max()}')
            return set(tmp[tmp['idx'] > self.idx]['idx'].values)
        return set()

    def get_areas(self):
        """
        Если интересует запрос по конкретной зоне (стране), то в параметры
        request нужно указать ID необходимой зоны, к примеру,
        для России: {'area': 113}
        :return: ДФ с зонами России
        """
        file_pkl = os.path.join(self.path_file, 'areas_113.pkl')
        if os.access(file_pkl, os.F_OK):
            self.print_read_msg(file_pkl)
            self.areas = pd.read_pickle(file_pkl)
        else:
            file_json = os.path.join(self.path_file, 'areas_113.json')
            if os.access(file_json, os.F_OK):
                data = self.read_json(file_json)
            else:
                data = self.download_page("https://api.hh.ru/areas/113")
            for element in data['areas']:
                # Если у зоны есть внутренние зоны
                if element['areas']:
                    for elem in element['areas']:
                        self.areas.loc[len(self.areas)] = [elem['id'],
                                                           elem['name']]
                # Если у зоны нет внутренних зон
                else:
                    self.areas.loc[len(self.areas)] = [element['id'],
                                                       element['name']]
            self.areas.to_pickle(file_pkl)
        self.areas.area_id = self.areas.area_id.astype(int)
        self.areas.sort_values('area_id', ignore_index=True, inplace=True)
        return self.areas

    def fill_na_open_vacancies(self):
        """
        Заполнение пропусков открытых вакансий у работодателя
        :return:
        """
        self.df['open_vacancies'].fillna(0, inplace=True)
        self.df['open_vacancies'] = self.df['open_vacancies'].astype(int)

    def save_df_to_csv(self, df=None):
        print('Сохраняю работодателей в .csv')
        # переименуем старый файл если он есть
        if os.access(self.file_csv, os.F_OK):
            old_files = os.path.join(self.path_to_save_files, 'old_employers')
            if not os.path.exists(old_files):
                os.makedirs(old_files)
            last_time = os.path.getmtime(self.file_csv)
            last_time = datetime.utcfromtimestamp(last_time)
            last_time = last_time.strftime("%Y-%m-%dT%H_%M_%S")
            old_file = os.path.basename(self.file_csv)
            old_file = old_file.replace('.csv', f'_{last_time}.csv')
            old_file = os.path.join(old_files, old_file)
            # если файл не был сохранен ранее - сохраним его
            if not os.access(old_file, os.F_OK):
                os.rename(self.file_csv, old_file)
        if df is not None:
            df.to_csv(self.file_csv, sep=';', index=False)
        else:
            self.fill_na_open_vacancies()
            self.df.to_csv(self.file_csv, sep=';', index=False)
        self.rows_in_df = len(self.df)
        print(f'Сохранено {self.rows_in_df} работодателей в .csv')

    def read_df_from_csv(self):
        if os.access(self.file_csv, os.F_OK):
            self.print_read_msg(self.file_csv)
            self.df = pd.read_csv(self.file_csv, sep=';')
            self.fill_na_open_vacancies()
            self.idx = self.df['employer_id'].max() + 1
            self.rows_in_df = len(self.df)

    def prepare_from_csv(self, start_id=None):
        """
        Чтение данных из .csv файлов: основной ДФ + из постраничного парсинга
        :param start_id: Начальный идентификатор работодателя
        :return: None
        """
        # id из основного ДФ
        self.read_df_from_csv()
        # id из постраничного парсинга
        self.read_saved_employers_pages()
        # id из экселевских файлов
        self.read_employers_from_excel()

        if start_id is not None:
            self.idx = start_id
            offset = 0
        else:
            offset = 200
        # сдвинем вниз идентификатор
        self.idx -= offset

        # Получим идентификаторы из постраничного парсинга
        idx_flt = self.df_json['employer_id'] > self.idx
        self.employer_ids = set(self.df_json[idx_flt]['employer_id'].values)
        # Получим идентификаторы из основного ДФ парсинга и добавим в множество
        idxs = self.df[self.df['employer_id'] > self.idx]['employer_id'].values
        self.employer_ids = self.employer_ids.union(set(idxs))
        # Добавим идентификаторы из лога индексов ненайденных работодателей
        self.employer_ids = self.employer_ids.union(self.read_idx_log())
        # Добавим идентификаторы из экселевских файлов, которые предоставили
        # для выполнения заданий по стажировке
        self.employer_ids = self.employer_ids.union(self.employer_ids_excel)

        # Прочитаем количество работодаталей в России с вакансиями
        params = dict(area=113, only_with_vacancies='true')
        data = self.download_page("https://api.hh.ru/employers", params)
        count_of_employers = data['found']
        print('Количество работодателей с вакансиями:', count_of_employers)

    def url_reader(self, idx_thread):
        """
        ЧИТАТЕЛЬ: здесь читаем и обрабатываем данные страниц
        :param idx_thread: номер потока
        :return: None
        """
        while True:
            # Проверяем, есть ли данные в очереди
            if self.files_queue.empty():
                print(f'Поток {idx_thread} завершен.')
                # выходим из цикла
                break
            # Получаем индекс работодателя из очереди
            idx = self.files_queue.get()
            # Скачиваем данные о работодателе
            employer = self.get_employer_from_idx(idx)
            # Помещаем данные в выходную очередь
            self.data_queue.put((idx, employer))

    def url_writer(self):
        """
        ПИСАТЕЛЬ
        :return: None
        """
        while True:
            # ожидаем получение данных
            if self.data_queue.empty():
                # !!!!! проверяем: живы ли потоки читателей url_reader() !!!!!
                if self.event_reader.is_set():
                    self.save_df_to_csv()
                    # если очередь пуста и все читатели завершили работу - ТО:
                    print_time(self.rq_time)
                    # конец работы - завершаем цикл
                    break
            else:
                # как только поступили данные извлекаем их и записываем в ДФ
                employer_id, employer = self.data_queue.get()

                if employer is None:
                    # запишем в лог обработанные индексы
                    with open(self.file_log, 'a+') as fw:
                        fw.write(f'{employer_id}\n')
                else:
                    # добавляем данные в итоговый ДФ
                    self.df.loc[len(self.df)] = [*employer]

                # Если данные были добавлены в ДФ - то через каждые 500 записей
                # или 2000 индексов сохраним ДФ в файл
                if len(self.df) > self.rows_in_df and (
                        not len(self.df) % 500 or not employer_id % 2000):
                    self.save_df_to_csv(self.df)
                # через каждые 300 записей или 1000 индексов заснем на 0.2 сек
                if not len(self.df) % 300 or not employer_id % 1000:
                    time.sleep(0.2)

    def make_queue_employer_ids(self, start_id=None, numbers_id=500000,
                                use_idx_from_excel=False):
        """
        Формирование очереди идентификаторов работодателей
        :param start_id: начальный индекс работодателя для парсинга hh.ru
        :param numbers_id: количество индексов для обработки
        :param use_idx_from_excel: использовать индексы из make_id_from_excel
        :return: None
        """
        self.prepare_from_csv()

        if start_id is not None:
            self.idx = start_id
            offset = 0
        else:
            offset = 200
        # добавим прочитанные данные со смещением, чтобы не пропустить индексы
        self.idx -= offset
        idxs = self.df[self.df['employer_id'] > self.idx]['employer_id'].values
        self.employer_ids = self.employer_ids.union(set(idxs))

        if use_idx_from_excel:
            range_idxs = self.employer_ids_excel
        else:
            range_idxs = range(self.idx, self.idx + numbers_id + offset)
        total = 0
        for idx in sorted(range_idxs):
            if idx not in self.employer_ids:
                self.files_queue.put(idx)
                total += 1
        print(f'Количество индексов для обработки: {total}')

    def get_employers_multi_old(self, num_threads=20,
                                start_id=None,
                                numbers_id=1000000,
                                use_idx_from_excel=False):
        """
        Загрузка работодателей в несколько потоков
        :param num_threads: Количество потоков
        :param start_id: начальный индекс работодателя для парсинга hh.ru
        :param numbers_id: количество индексов для обработки
        :param use_idx_from_excel: использовать индексы из make_id_from_excel
        :return: датафрейм
        """
        # !!!!! включаем управление событиями !!!!!
        self.event_reader = threading.Event()
        self.rq_time = time.time()

        self.prepare_from_csv()

        if start_id is not None:
            self.idx = start_id
            offset = 0
        else:
            offset = 200
        # добавим прочитанные данные со смещением, чтобы не пропустить индексы
        self.idx -= offset
        idxs = self.df[self.df['employer_id'] > self.idx]['employer_id'].values
        self.employer_ids = self.employer_ids.union(set(idxs))

        if use_idx_from_excel:
            range_idxs = self.employer_ids_excel
        else:
            range_idxs = range(self.idx, self.idx + numbers_id + offset)
        for idx in range_idxs:
            if idx not in self.employer_ids:
                self.files_queue.put(idx)

        if self.files_queue.empty():
            print('НЕТ страниц для обработки.')
        else:
            # пишем в 1 поток. Если данные писать в несколько потоков,
            # то нужно еще использовать блокировщик threading.Lock()
            twr = threading.Thread(target=self.url_writer)
            twr.start()

            # читаем и обрабатываем в Х потоков
            threads = []
            for idx_thread in range(num_threads):
                thread = threading.Thread(target=self.url_reader,
                                          args=(idx_thread + 1,))
                threads.append(thread)
                thread.start()

            # ждем, когда все файлы прочитаются
            [thread.join() for thread in threads]
            # как все потоки reader() завершены
            # !!!!! скажем об этом writer() !!!!!
            self.event_reader.set()
        return self.df

    def get_employer_from_idx(self, idx):
        """
        Получение работодателя со страницы
        :param idx: ID работодателя
        :return:
        """
        url = f"https://api.hh.ru/employers/{idx}"
        # print(url)
        data = self.download_page(url)
        if 'id' in data:
            employer = [data['area']['id'], data['id'], data['name'],
                        data['open_vacancies']]
            print(employer)
            return employer

    def get_employers_page(self, area=113, employer_type=None, page=0):
        """
        Скачиваем страницу с работодателями
        :param area: регион
        :param employer_type: тип компании
        :param page: номер страницы
        :return: содержимое страницы .json()
        """
        params = {'area': area,
                  'type': employer_type,
                  'only_with_vacancies': 'true',
                  'per_page': 100,
                  'page': page}
        data = self.download_page("https://api.hh.ru/employers", params)
        return data

    def area_reader(self, idx_thread):
        """
        ЧИТАТЕЛЬ: здесь читаем и обрабатываем данные страниц
        :param idx_thread: номер потока
        :return: None
        """
        while True:
            # Проверяем, есть ли данные в очереди
            if self.files_queue.empty():
                print(f'Поток {idx_thread} завершен.')
                # выходим из цикла
                break
            # Получаем регион из очереди
            area_id, area_name = self.files_queue.get()
            print(f'Регион: {area_id} - {area_name}')

            # Скачиваем данные о работодателях одного региона
            self.get_employers_from_area(area_id, area_name)
            # Помещаем данные в выходную очередь
            self.data_queue.put((area_id, area_name))

    def area_writer(self):
        """
        ПИСАТЕЛЬ
        :return: None
        """
        while True:
            # ожидаем получение данных
            if self.data_queue.empty():
                # !!!!! проверяем: живы ли потоки читателей url_reader() !!!!!
                if self.event_reader.is_set():
                    # если очередь пуста и все читатели завершили работу - ТО:
                    print_time(self.rq_time)
                    # конец работы - завершаем цикл
                    break
            else:
                # запишем в лог обработанные индексы регионов
                area_id, area_name = self.data_queue.get()
                json_log = os.path.join(self.path_to_save_files,
                                        'employers_pages.log')
                with open(json_log, 'a+') as fw:
                    fw.write(f'{area_id}\n')

    def make_queue_areas_ids(self, areas_idxs=[]):
        """
        Формирование очереди идентификаторов регионов
        :param areas_idxs: Список area_id
        :return: None
        """
        areas = self.get_areas()
        if areas_idxs:
            areas = areas[areas.area_id.isin(areas_idxs)]
        total = 0
        for row in areas.itertuples(index=False):
            self.files_queue.put((row.area_id, row.area_name))
            total += 1
        print(f'Количество регионов для обработки: {total}')
        data = self.get_employers_page()
        total_pages = data['pages']
        print('Количество работодателей с вакансиями:', data['found'])
        print('Количество страниц:', total_pages)

    def get_employers_from_area(self, area_id, area_name):
        print(f'Регион: {area_id} - {area_name}')
        data = self.get_employers_page(area=area_id)
        total_pages = data['pages']
        employer_types = (None,)
        if total_pages > 50:
            employer_types = ('company', 'agency', 'project_director',
                              'private_recruiter')
        for employer_type in employer_types:
            data = self.get_employers_page(area=area_id,
                                           employer_type=employer_type)
            total_pages = data['pages']
            print('Количество работодателей с вакансиями:', data['found'],
                  'Количество страниц:', total_pages)
            e_type = ''
            if employer_type is not None:
                e_type = f'_{employer_type}'

            if total_pages > 50:
                total_pages = 50
            for page in range(total_pages):
                print(f'Обрабатываю страницу: {page}')
                data = self.get_employers_page(area=area_id, page=page)
                name_file = f'area{area_id:04}_page{page:03}_employer{e_type}.json'
                file_json = os.path.join(self.json_path, name_file)
                with open(file_json, 'w') as outfile:
                    json.dump(data, outfile)

    def get_all_employers_pages_multi(self, num_threads=13, areas_idxs=[]):
        """
        Многопоточная загрузка работодателей по регионам
        с сохранением их в файлы .json
        :param num_threads: Количество потоков
        :param areas_idxs: список регионов для обработки
        :return: None
        """
        # Вызов функции управления потоками и передача в неё параметров
        self.multithreaded_processing(num_threads,
                                      (self.make_queue_areas_ids, areas_idxs),
                                      self.area_reader,
                                      self.area_writer)

    def get_all_employers_pages(self, areas_idxs=[]):
        """
        Загрузка работодателей по регионам с сохранением их в файлы .json
        :param areas_idxs: список регионов для обработки
        :return: None
        """
        # Формирование очереди идентификаторов регионов
        self.make_queue_areas_ids(areas_idxs=areas_idxs)
        # Парсинг страниц работодателей по регионам в один поток
        self.area_reader(0)
        # На всякий случай очистка очередей
        self.data_queue.queue.clear()
        self.files_queue.queue.clear()

    def file_writer(self):
        """
        ПИСАТЕЛЬ
        :return: None
        """
        while True:
            # ожидаем получение данных
            if self.data_queue.empty():
                # !!!!! проверяем: живы ли потоки читателей file_reader() !!!!!
                if self.event_reader.is_set():
                    self.df_json.to_csv(self.json_to_csv, sep=';', index=False)
                    # если очередь пуста и все читатели завершили работу - ТО:
                    print('Все файлы объединены.')
                    print_time(self.rq_time)
                    # конец работы - завершаем цикл
                    break
            else:
                # как только поступили данные извлекаем их и записываем в ДФ
                data = self.data_queue.get()
                # добавляем данные в итоговый ДФ
                if len(self.df_json.index):
                    self.df_json = pd.concat([self.df_json, data],
                                             ignore_index=True)
                else:
                    self.df_json = data
                print(f'В датафрейме {len(self.df_json.index)} записей')

    def file_reader(self, idx_thread):
        """
        ЧИТАТЕЛЬ: здесь читаем и обрабатываем данные файлов
        :param idx_thread: номер потока
        :return: None
        """
        # здесь читаем и обрабатываем данные файлов
        while True:
            # Проверяем, есть ли файлы в очереди
            if self.files_queue.empty():
                print(f'Поток {idx_thread} завершен.')
                # выходим из цикла
                break
            # Получаем имя файла из очереди
            file_json = self.files_queue.get()
            file = os.path.basename(file_json)
            area_id = file.split('_')[0][4:]
            data = self.read_json(file_json)
            df = pd.DataFrame(columns=self.df.columns)
            for item in data['items']:
                employer = [area_id, item['id'], item['name'],
                            item['open_vacancies']]
                df.loc[len(df)] = [*employer]
            self.data_queue.put(df)

    def read_saved_employers_pages(self):
        """
        Чтение итогового файла с работодателями,
        полученного из read_employers_json_pages
        :return: None
        """
        if os.access(self.json_to_csv, os.F_OK):
            self.print_read_msg(self.json_to_csv)
            self.df_json = pd.read_csv(self.json_to_csv, sep=';')

    def make_queue_files(self):
        for file in os.listdir(self.json_path):
            file_json = os.path.join(self.json_path, file)
            if file.endswith('.json') and os.path.getsize(file_json) > 65:
                # создаем и заполняем очередь именами файлов
                self.files_queue.put(file_json)

    def read_employers_json_pages(self, num_threads=5):
        """
        Многопоточная загрузка работодателей по регионам
        из файлов .json
        :param num_threads: Количество потоков
        :return: None
        """
        # Вызов функции управления потоками и передача в неё параметров
        self.multithreaded_processing(num_threads,
                                      (self.make_queue_files,),
                                      self.file_reader,
                                      self.file_writer)

    def get_vacancies_page(self, employer_id, area=113, page=0,
                           date_from=None, date_to=None):
        url = 'https://api.hh.ru/vacancies'
        params = {'area': area,
                  'employer_id': employer_id,
                  'per_page': 100,
                  'page': page,
                  'date_from': date_from,
                  'date_to': date_to,
                  }
        data = self.download_page(url, params)
        return data

    def get_vacancies_pages(self, employer_id, area=113,
                            date_from=None, date_to=None, data=None):
        """
        Загрузка всех вакансий работодателя
        :param employer_id: id работодателя
        :param area: регион
        :param date_from: дата с
        :param date_to: дата по
        :param data: данные, загруженные с нулевой страницы
        :return: ДФ с данными по вакансиям со всех страниц
        """
        if data is None:
            data = self.get_vacancies_page(employer_id, area=area,
                                           date_from=date_from,
                                           date_to=date_to)
        total_pages = data['pages']

        if total_pages > 20:
            total_pages = 20
        df = pd.DataFrame(columns=self.dict_keys + ('snippet', 'url'))

        vacancy_exist = set()
        if (self.checking_existing_vacancies or
                self.load_missing_vacancy_fields):
            # проверка наличия вакансии в уже скачанных вакансий - достаем ДФ
            name_attribute = f'df_{employer_id}'
            if hasattr(ParsingHH, name_attribute):
                vacancy_exist = getattr(ParsingHH, name_attribute)
                if self.checking_existing_vacancies:
                    vacancy_exist = set(vacancy_exist['id'].to_list())
                elif self.load_missing_vacancy_fields:
                    flt = vacancy_exist[self.keys_not_item].isna().all(axis=1)
                    vacancy_exist = set(vacancy_exist[flt]['id'].to_list())

        for page in range(total_pages):
            data = self.get_vacancies_page(employer_id, area=area,
                                           page=page,
                                           date_from=date_from,
                                           date_to=date_to)
            for item in data.get('items', []):
                # проверка наличия вакансии в уже скачанных данных
                if self.checking_existing_vacancies:
                    if str(item.get('id', '')) in vacancy_exist:
                        continue
                elif self.load_missing_vacancy_fields:
                    if str(item.get('id', '')) not in vacancy_exist:
                        continue

                url = item['url']
                snippet = item.get('snippet', dict())

                # если не было ошибки про требование ввода капчи -->
                # читаем саму вакансию
                if not self.errors_captcha_required:
                    print(url)
                    enriched = self.download_page(url)
                    if enriched.get('errors', None) is None:
                        item = enriched
                    else:
                        if enriched.get('description', None) is None:
                            # словили ошибку про требование ввода капчи->ставим
                            # глобальный флаг, чтобы больше не читать вакансии
                            self.errors_captcha_required = True
                        else:
                            # получили ошибку description": "Not Found"
                            item = None
                if (self.errors_captcha_required and
                        self.load_missing_vacancy_fields):
                    # если загрузка отсутствующих полей и словили капчу - выход
                    break
                if item is not None:
                    df.loc[len(df)] = [*[item.get(key) for key
                                         in self.dict_keys], snippet, url]
        return df

    @staticmethod
    def date_to_str(date):
        return date.strftime('%Y-%m-%d')

    def get_employer_all_vacancies(self, employer_id, area=113,
                                   date_from=None, date_to=None, days_off=64):
        """
        Получение списка вакансий работодателя
        :param employer_id: id работодателя
        :param area: регион
        :param date_from: дата с
        :param date_to: дата по
        :param days_off: диапазон в днях между датами
        :return: количество найденных вакансий
        """
        date_now = datetime.now()
        if date_from is None:
            date_from = date_now - timedelta(days=days_off - 1)
        if date_to is None:
            date_to = date_now

        data = self.get_vacancies_page(employer_id, area=area,
                                       date_from=self.date_to_str(date_from),
                                       date_to=self.date_to_str(date_to))

        found_vacancies = data.get('found', 0)

        print(f'Период: {days_off:>3} дн., {self.date_to_str(date_from)} - '
              f'{self.date_to_str(date_to)} '
              f'вакансий: {found_vacancies}')

        if 0 < found_vacancies < 2001 or days_off < 2:
            # добавление данных по вакансиям в ДФ одного работодателя -->
            # в функцию передать данные первой страницы
            date_from = self.date_to_str(date_from)
            date_to = self.date_to_str(date_to)
            df_temp = self.get_vacancies_pages(employer_id,
                                               area=area,
                                               date_from=date_from,
                                               date_to=date_to,
                                               data=data)
            # атрибут класса - ДФ,
            # в который собираются данные одного работодателя
            name_attribute = f'df_{employer_id}'
            if not hasattr(ParsingHH, name_attribute):
                setattr(ParsingHH, name_attribute, pd.DataFrame())
            vdf = getattr(ParsingHH, name_attribute)
            if self.load_missing_vacancy_fields:
                # замена существующих вакансий скачанными --> оставим в ДФ
                # только id вакансий, которые не подлежат замене
                vacancy_exist = df_temp['id'].to_list()
                vdf = vdf[~vdf['id'].isin(vacancy_exist)]
            if len(vdf):
                vdf = pd.concat([vdf, df_temp], ignore_index=True)
            else:
                vdf = df_temp
            setattr(ParsingHH, name_attribute, vdf)

        elif found_vacancies > 2000:
            # делим период пополам и смотрим, что получилось
            days_off //= 2
            # первый отрезок дат
            date_to1 = date_from + timedelta(days=days_off - 1)
            self.get_employer_all_vacancies(employer_id,
                                            area=area,
                                            date_from=date_from,
                                            date_to=date_to1,
                                            days_off=days_off)
            # второй отрезок дат
            date_fr2 = date_to - timedelta(days=days_off - 1)
            self.get_employer_all_vacancies(employer_id,
                                            area=area,
                                            date_from=date_fr2,
                                            date_to=date_to,
                                            days_off=days_off)
        return found_vacancies

    def all_employers_to_df(self):
        # ДФ из основного ДФ
        self.read_df_from_csv()
        print(self.df.info())
        # ДФ из постраничного парсинга
        self.read_saved_employers_pages()
        print(self.df_json.info())
        # ДФ из экселевских файлов
        self.read_employers_from_excel()
        print(self.df_excel.info())
        df = pd.concat([self.df, self.df_json, self.df_excel])
        df.employer_name.fillna('', inplace=True)
        df.drop_duplicates(inplace=True, ignore_index=True)
        return df

    def make_file_emp_df_csv(self, employer_id):
        return os.path.join(self.parsed_employers, f'{employer_id:08}.csv')

    def load_employer_vacancies(self, employer_id):
        file_emp_df_csv = self.make_file_emp_df_csv(employer_id)
        if os.access(file_emp_df_csv, os.F_OK):
            tmp = pd.read_csv(file_emp_df_csv, sep=';')
            tmp.dropna(subset=['id', 'name'], axis=0, inplace=True)
            tmp['id'] = tmp['id'].apply(lambda x: str(int(x)))
            if 'url' not in tmp.columns:
                tmp['url'] = ''
            return tmp
        return pd.DataFrame()

    def employer_vacancies_reader(self, idx_thread):
        """
        ЧИТАТЕЛЬ: Получение всех вакансий работодателя
        :param idx_thread: номер потока
        :return: None
        """
        while True:
            # Проверяем, есть ли данные в очереди
            if self.files_queue.empty():
                print(f'Поток {idx_thread} завершен.')
                # выходим из цикла
                break
            # Получаем индекс и наименование работодателя из очереди
            employer_id, employer_name = self.files_queue.get()
            print(f'Работодатель: id={employer_id} {employer_name[:50]}')

            name_attribute = f'df_{employer_id}'
            # проверять ранее загруженные файлы с вакансиями
            if (self.checking_existing_vacancies or
                    self.load_missing_vacancy_fields):
                emp_df = self.load_employer_vacancies(employer_id)
            else:
                emp_df = pd.DataFrame()
            # установим атрибут класса ДФ, в который будем собирать данные
            setattr(ParsingHH, name_attribute, emp_df)
            # Скачиваем данные о вакансиях работодателя
            found_ = self.get_employer_all_vacancies(employer_id)
            emp_df = getattr(ParsingHH, name_attribute)
            # удалим атрибут класса (ДФ уже не нужен)
            delattr(ParsingHH, name_attribute)

            # Если вакансии найдены:
            if (len(emp_df) == found_ or self.checking_existing_vacancies or
                    self.load_missing_vacancy_fields):
                if len(emp_df):
                    # сохраняем в .csv если были найдены вакансии
                    file_emp_df_csv = self.make_file_emp_df_csv(employer_id)
                    emp_df.to_csv(file_emp_df_csv, sep=';', index=False)

                # Помещаем данные в выходную очередь
                self.data_queue.put((employer_id, emp_df))
            else:
                print(f'Ошибка получения данных employer_id:{employer_id}')
                # ошибка при получении данных --> запишем в лог employer_id
                # обработанного работодателя
                with open(self.parsed_employers_log.replace('.log', '.errors'),
                          'a+') as fw:
                    fw.write(f'{employer_id};{found_}\n')
            if (self.errors_captcha_required and
                    self.load_missing_vacancy_fields):
                # Если выполняем загрузку отсутствующих полей и словили капчу:
                # очистка очереди и выход, т.к. не имеет смысла гонять цикл
                self.files_queue.queue.clear()

    def load_df_vacancies_from_csv(self):
        if os.access(self.file_vacancies, os.F_OK):
            self.df_vacancies = pd.read_csv(self.file_vacancies, sep=';')
            self.df_vacancies.dropna(subset=['id'], axis=0, inplace=True)
            self.df_vacancies['id'] = self.df_vacancies['id'].astype(int)
            self.df_vacancies.drop_duplicates(subset=['id'], keep='last',
                                              ignore_index=True,
                                              inplace=True)
            self.rows_in_vacancies = len(self.df_vacancies)

    def save_df_vacancies_to_csv(self):
        print('Сохраняю вакансии в .csv')
        # переименуем старый файл если он есть
        if os.access(self.file_vacancies, os.F_OK):
            old_vacancies_files = os.path.join(self.path_to_save_files,
                                               'old_vacancies')
            if not os.path.exists(old_vacancies_files):
                os.makedirs(old_vacancies_files)
            last_time = os.path.getmtime(self.file_vacancies)
            last_time = datetime.utcfromtimestamp(last_time)
            last_time = last_time.strftime("%Y-%m-%dT%H_%M_%S")
            old_file = os.path.basename(self.file_vacancies)
            old_file = old_file.replace('.csv', f'_{last_time}.csv')
            old_file = os.path.join(old_vacancies_files, old_file)
            # если файл не был сохранен ранее - сохраним его
            if not os.access(old_file, os.F_OK):
                os.rename(self.file_vacancies, old_file)
        self.df_vacancies.to_csv(self.file_vacancies, sep=';', index=False)
        self.rows_in_vacancies = len(self.df_vacancies)
        print(f'Сохранено {self.rows_in_vacancies} вакансий в .csv')

    def employer_vacancies_writer(self):
        """
        ПИСАТЕЛЬ
        :return: None
        """
        while True:
            # ожидаем получение данных
            if self.data_queue.empty():
                # !!!!! проверяем: живы ли потоки читателей _reader() !!!!!
                if self.event_reader.is_set():
                    self.save_df_vacancies_to_csv()
                    # если очередь пуста и все читатели завершили работу - ТО:
                    print_time(self.rq_time)
                    # конец работы - завершаем цикл
                    break
            else:
                # как только поступили данные извлекаем их и записываем в ДФ
                employer_id, emp_df = self.data_queue.get()

                # запишем в лог employer_id обработанного работодателя
                with open(self.parsed_employers_log, 'a+') as fw:
                    fw.write(f'{employer_id}\n')

                # добавляем данные в итоговый ДФ
                self.df_vacancies = pd.concat([self.df_vacancies, emp_df],
                                              ignore_index=True)

                # Если вакансий добавлено более 2000 записей сохраним ДФ
                if len(self.df_vacancies) - self.rows_in_vacancies > 2000:
                    self.save_df_vacancies_to_csv()

    def employers_vacancies_multi(self, num_threads=13, idxs_employers=None,
                                  max_employers=None):
        """
        Многопоточная загрузка вакансий работодателей
        :param num_threads: Количество потоков
        :param idxs_employers: Множество идентификаторов работодателей
        :param max_employers: Количество работодаталей для парсинга
        :return: None
        """
        self.load_df_vacancies_from_csv()
        # Вызов функции управления потоками и передача в неё параметров
        self.multithreaded_processing(num_threads,
                                      (self.make_queue_employers,
                                       idxs_employers, max_employers),
                                      self.employer_vacancies_reader,
                                      self.employer_vacancies_writer)

    def read_parsed_employers_log(self):
        """
        Чтение из лог-файла идентификаторов обработанных работодателей
        :return: множество идентификаторов
        """
        if os.access(self.parsed_employers_log, os.F_OK):
            tmp = pd.read_csv(self.parsed_employers_log, header=None)
            tmp.columns = ['idx']
            return set(tmp['idx'].values)
        return set()

    def make_queue_employers(self, idxs_employers=None, max_employers=None):
        """
        Формирование очереди работодателей для парсинга вакансий
        :param idxs_employers: Множество идентификаторов работодателей
        :param max_employers: Количество работодаталей для парсинга
        :return:
        """
        # загрузка регионов России
        areas = self.get_areas()
        # Чтение данных из основного ДФ
        self.read_df_from_csv()
        # фильтрация Российских работодателей
        ru_hh = self.df[self.df.area_id.isin(areas.area_id)]
        print('Кол-во работодателей в России:', len(ru_hh))
        if idxs_employers is not None:
            # парсим только переданных работодателей
            ru_open_vacancies = ru_hh[ru_hh.employer_id.isin(idxs_employers)]
            log_idx = set()
        else:
            # парсим только работодателей c вакансиями
            ru_open_vacancies = ru_hh[ru_hh.open_vacancies > 0]
            # Прочитаем идентификаторы из лога обработанных работодателей
            log_idx = self.read_parsed_employers_log()
        print('Из них с открытыми вакансиями:', len(ru_open_vacancies))
        total = 0
        for row in ru_open_vacancies.itertuples(index=False):
            if row.employer_id not in log_idx:
                self.files_queue.put((row.employer_id, row.employer_name))
                total += 1
                if max_employers is not None and total > max_employers:
                    break
        print(f'Количество работодателей для обработки: {total}')

    def make_patterns(self, return_category=False, only_IT=False):
        """
        Формирование списка вакансий для мониторинга
        :param return_category: вернуть список категорий
        :param only_IT: Отобрать только шаблоны с пометкой "ИТ"
        :return: список вакансий
        """

        def join_names(row):
            """
            Объединение столбцов с вакансиями в одну строку для запроса
            :param row: строка ДФ
            :return: вакансия для поиска
            """
            out_name = row.name1.strip()
            if not pd.isna(row.name2):
                out_name = f'({out_name.strip()}) OR ({row.name2.strip()})'
            if not pd.isna(row.name3):
                out_name = f'{out_name} OR ({row.name3.strip()})'
            return out_name

        # ДФ со списком наименований вакансий
        df_names = pd.read_excel(os.path.join(self.path_file, 'patterns.xlsx'))
        df_names['names'] = df_names.apply(join_names, axis=1)
        df_names['is_it'] = df_names['is_it'].apply(
            lambda x: False if pd.isna(x) or not x else True)
        if only_IT:
            df_names = df_names[df_names.is_it]
        df_names.to_excel(os.path.join(self.path_file, 'make_patterns.xlsx'),
                          index=False)
        for row in df_names.itertuples(index=False):
            self.patterns_dict[row.names] = row.is_it
        return self.patterns_dict if return_category + only_IT < 2 else [True]

    def get_selected_vacancy_info(self, area=113, **kwargs):
        """
        Получение количества вакансий по шаблону
        :param area: регион
        :param kwargs: шаблон поиска вакансий
        :return: результат запроса
        """
        url = 'https://api.hh.ru/vacancies'
        params = {'area': area,
                  'enable_snippets': 'false',
                  # 'text': text,
                  'per_page': 10
                  }
        for key, val in kwargs.items():
            params[key] = val

        data = dict()
        data = self.download_page(url, params)
        return data

    def getting_numbers_vacancy_resume(self, kind='vacancy', search='text'):
        """
        Загрузка информации о количестве вакансий из шаблонов
        Сохранение в .csv и .xlsx
        :param kind: вид поиска: вакасии/резюме
        :param search: где искать
        :return: None
        """
        file_name = os.path.join(self.path_to_save_files, 'job_monitoring')
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        name_fields = {'text': [f'{kind}_text', 'is_it'],
                       'specialization': ['id', search],
                       'professional_role': ['id', search]}
        ins_fields = name_fields[search]
        tmp = pd.DataFrame(columns=['date', *ins_fields, f'{kind}_counts'])
        # print(tmp.columns)

        date_now = datetime.now()

        print('Поиск вакансий / резюме ...')
        for pattern, item in self.funcs[search]().items():
            print(f'Запрос по шаблону: {search}={pattern}')
            data = dict()
            if kind == 'vacancy':
                data = self.get_selected_vacancy_info(**{search: pattern})
            else:
                data = self.get_selected_resume_info(**{search: pattern})
            tmp.loc[len(tmp)] = [date_now, pattern, item, data.get('found', 0)]
        tmp.sort_values([f'{kind}_counts', *ins_fields], ascending=False,
                        inplace=True)
        date_now = date_now.strftime("%Y-%m-%dT%H_%M_%S")
        file_name = os.path.join(file_name,
                                 f'{kind}_{search[:4]}_{date_now}.csv')
        tmp.to_csv(file_name, sep=';', index=False)
        tmp.to_excel(file_name.replace('.csv', '.xlsx'), index=False)

    def get_selected_resume_info(self, area=113, **kwargs):
        """
        Получение количества резюме по шаблону
        :param area: регион
        :param kwargs: шаблон поиска резюме
        :return: результат запроса
        """
        # https://hh.ru/search/resume?area=113
        # &employment=full
        # &job_search_status=active_search&job_search_status=looking_for_offers
        # &relocation=living_or_relocation
        # &gender=unknown
        # &logic=normal
        # &pos=full_text
        # &exp_period=all_time
        # &search_period=0
        # &text=python
        url = 'https://hh.ru/search/resume'
        params = {'area': area,
                  'employment': 'full',
                  'job_search_status': ['active_search', 'looking_for_offers'],
                  'relocation': 'living_or_relocation',
                  'logic': 'normal',
                  'gender': 'unknown',
                  'pos': 'full_text',
                  'exp_period': 'all_time',
                  'search_period': 0,
                  # 'text': text,
                  'per_page': 10
                  }
        for key, val in kwargs.items():
            params[key] = val

        data = dict()
        html = self.download_page(url, params, ret_json=False)
        html_text = html.content

        # # чтение файла для опытов
        # step_obj = MakeHTML(base_dir=self.path_to_save_files)
        # html_text = step_obj.read_html_file('поиск_резюме.html')

        soup = BeautifulSoup(html_text, 'lxml')
        result = soup.select('.novafilters-list__item .bloko-text_tertiary')
        if result:
            counts = ''.join(re.findall('\d+', result[0].text))
            data['found'] = int(counts)
        return data

    def getting_number_resumes(self):
        """
        Загрузка информации по количеству резюме из шаблонов
        Сохранение в .csv и .xlsx
        :return: None
        """
        file_name = os.path.join(self.path_to_save_files, 'resume_monitoring')
        if not os.path.exists(file_name):
            os.makedirs(file_name)
        tmp = pd.DataFrame(columns=['date', 'resume_name', 'resume_counts'])
        date_now = datetime.now()
        self.make_patterns()
        print('Поиск резюме...')
        for pattern in self.patterns_dict:
            print(f'Запрос по шаблону "{pattern}"')
            data = self.get_selected_resume_info(text=pattern)
            tmp.loc[len(tmp)] = [date_now, pattern, data.get('found', 0)]
        tmp.sort_values(['resume_counts', 'resume_name'], ascending=False,
                        inplace=True)
        date_now = date_now.strftime("%Y-%m-%dT%H_%M_%S")
        file_name = os.path.join(file_name, f'number_resumes_{date_now}.csv')
        tmp.to_csv(file_name, sep=';', index=False)
        tmp.to_excel(file_name.replace('.csv', '.xlsx'), index=False)

    def get_id_name_from_dict(self, row, kind_dict=1):
        """
        Заполнение справочников-словарей специализаций/проф.ролей
        :param row: строка из .json
        :param kind_dict: 1-специализаци, другое проф.роли
        :return: row_id, row_name
        """
        row_id, row_name = row.get('id', None), row.get('name', None)
        if row_id is not None:
            if kind_dict == 1:
                self.specializations_dict[row_id] = row_name
            else:
                self.professional_roles_dict[row_id] = row_name
        return str(row_id), row_name

    def make_dict_specializations(self, return_category=False,
                                  only_IT=False):
        """
        Справочник специализаций
        :param return_category: вернуть список категорий для корневой
        :param only_IT: отобразить только категории "ИТ"
        :return:
        """
        # '1' - Информационные технологии, интернет, телеком
        file_json = os.path.join(self.path_to_save_files,
                                 'specializations.json')
        if os.access(file_json, os.F_OK):
            data = self.read_json(file_json)
        else:
            data = self.download_page('https://api.hh.ru/specializations')
        self.category_specializations = dict()
        for row in data:
            row_id, row_name = self.get_id_name_from_dict(row)
            if only_IT and row_id not in ['1']:
                continue
            category_list = [row_id]
            for item in row.get('specializations', []):
                item_id, _ = self.get_id_name_from_dict(item)
                category_list.append(item_id)
            self.category_specializations[row_id] = (row_name, category_list)
        if not return_category:
            return self.specializations_dict
        if not only_IT:
            return self.category_specializations
        return self.category_specializations['1'][1]

    def make_dict_professional_roles(self, return_category=False,
                                     only_IT=False):
        """
        Справочник профессиональных ролей
        :param return_category: вернуть список категорий для корневой
        :param only_IT: отобразить только категории "ИТ"
        :return:
        """
        # '11' - Информационные технологии
        file_json = os.path.join(self.path_to_save_files,
                                 'professional_roles.json')
        if os.access(file_json, os.F_OK):
            data = self.read_json(file_json)
        else:
            data = self.download_page('https://api.hh.ru/professional_roles')
        self.category_prof_roles = dict()
        for row in data.get('categories', []):
            row_id, row_name = self.get_id_name_from_dict(row, kind_dict=2)
            if only_IT and row_id not in ['11']:
                continue
            category_list = [row_id]
            for item in row.get('roles', []):
                item_id, _ = self.get_id_name_from_dict(item, kind_dict=2)
                category_list.append(item_id)
            self.category_prof_roles[row_id] = (row_name, category_list)
        if not return_category:
            return self.professional_roles_dict
        if not only_IT:
            return self.category_prof_roles
        return self.category_prof_roles['11'][1]

    def files_from_job_monitoring_to_df(self, kind='vacancy',
                                        search='specialization',
                                        type_files='.csv',
                                        only_IT=False):
        """
        Чтение файлов с информацией о количестве вакансий / резюме из шаблонов
        :param kind: вид поиска: вакасии / резюме
        :param search: где искать
        :param type_files: тип файлов
        :param only_IT: отобрать только профессии, относящиеся к ИТ
        :return: ДФ
        """
        df = pd.DataFrame()
        files_path = Path(self.path_to_save_files).joinpath('job_monitoring')
        for file in files_path.glob(f'{kind}_{search[:4]}_*{type_files}'):
            self.print_read_msg(file)
            if type_files == '.xlsx':
                tmp = pd.read_excel(file, dtype=str)
            else:
                tmp = pd.read_csv(file, sep=';', dtype=str)

            tmp['date'] = pd.to_datetime(tmp['date']).dt.date
            if search == 'text':
                if 'is_it' in tmp.columns:
                    tmp.rename(columns={'is_it': 'id'}, inplace=True)
                if 'resume_text' in tmp.columns:
                    tmp.rename(columns={'resume_text': 'vacancy_text'},
                               inplace=True)
                tmp['id'] = tmp['id'].astype(bool)

            # фильтрация ФД
            if only_IT and self.filter_it:
                tmp = tmp[tmp['id'].isin(self.filter_it)]

            # преобразование количества в целый тип
            for col_name in tmp.columns:
                if col_name.endswith('counts'):
                    tmp[col_name] = tmp[col_name].astype(int)

            if not len(df):
                df = tmp
            else:
                df = pd.concat([df, tmp], ignore_index=True)
        df.drop_duplicates(df.columns[:3], keep='last', inplace=True)
        return df

    def make_spec_to_category(self):
        # получаем специализации по категориям
        self.make_dict_specializations(return_category=True)
        self.spec_to_category = dict()
        for key, val in self.category_specializations.items():
            for spec in val[1]:
                self.spec_to_category[spec] = f'{key.rjust(2, "0")}-{val[0]}'

    def join_vacancy_resume(self, search='specialization', type_files='.csv',
                            only_IT=False):
        """
        Объединение ДФ с информацией о количестве вакансий / резюме
        :param search: где искать
        :param type_files: тип файлов
        :param only_IT: отобрать только профессии, относящиеся к ИТ
        :return: ДФ merge, ДФ concat
        """
        # формирование фильтра для отбора профессий, относящихся к ИТ
        if only_IT:
            self.filter_it = set(self.funcs[search](return_category=True,
                                                    only_IT=True))

        # получаем список из двух ДФ с данными по вакансиям и резюме
        df_vr = [self.files_from_job_monitoring_to_df(kind=kind,
                                                      search=search,
                                                      type_files=type_files,
                                                      only_IT=only_IT)
                 for kind in ('vacancy', 'resume')]
        # объединяем эти ДФ merge
        df_merge = df_vr[0].merge(df_vr[1],
                                  on=df_vr[0].columns.values.tolist()[:3],
                                  how='outer')
        # объединяем эти ДФ concat
        df_columns = 'date;id;specialization;counts'.split(';')
        df_concat = df_vr[0]
        df_concat.columns = df_columns
        df_concat['kind'] = 'Вакансии'
        df = df_vr[1]
        df.columns = df_columns
        df['kind'] = 'Резюме'
        df_concat = pd.concat([df_concat, df], ignore_index=True)

        # добавляем категорию в ДФ
        self.make_spec_to_category()
        df_merge['category'] = df_merge['id'].map(self.spec_to_category)
        df_concat['category'] = df_concat['id'].map(self.spec_to_category)
        # добавление в каждую строку информацию о вакансиях и резюме
        df_columns = df_merge.columns.values.tolist()[:-1]
        df_concat = df_concat.merge(df_merge[df_columns],
                                    on=df_columns[:3],
                                    how='left')
        return df_merge, df_concat

    def make_df_vacancies_from_csv(self):
        self.load_df_vacancies_from_csv()
        files_path = Path(self.parsed_employers)
        pbar = tqdm(total=sum(1 for _ in files_path.glob('*.csv')))
        for file in sorted(files_path.glob('*.csv')):
            # self.print_read_msg(file)
            pbar.update(1)
            # если файл не пустой
            if file.stat().st_size > 200:
                df = pd.read_csv(file, sep=';')
                df.dropna(subset=['id', 'name'], axis=0, inplace=True)
                # Если ДФ не пустой
                if len(df):
                    df['id'] = df['id'].astype(int)
                    if 'url' not in df.columns:
                        df['url'] = ''
                    if not len(self.df_vacancies):
                        self.df_vacancies = df
                    else:
                        self.df_vacancies = pd.concat([self.df_vacancies, df])
        pbar.close()
        # удалим дубликаты вакансий
        self.df_vacancies['id'] = self.df_vacancies['id'].astype(int)
        self.df_vacancies.drop_duplicates(subset=['id'], keep='last',
                                          ignore_index=True,
                                          inplace=True)
        self.save_df_vacancies_to_csv()

    def make_bad_df_vacancies_from_csv(self):
        idxs = set()
        files_path = Path(self.parsed_employers)
        for file in files_path.glob('[01]*.csv'):
            # если файл не пустой
            if file.stat().st_size < 200:
                idx = int(file.name.split('.')[0])
                data = self.get_vacancies_page(idx)
                found = data.get('found', 0)
                if not found:
                    file.unlink()
                else:
                    idxs.add(idx)
        return idxs

    def doubtful_df_vacancies_from_csv(self):
        idxs = set()
        files_path = Path(self.parsed_employers)
        for file in files_path.glob('[01]*.csv'):
            # если файл не пустой
            if file.stat().st_size > 200:
                df = pd.read_csv(file, sep=';')
                len_df = len(df)
                df.dropna(subset=['id', 'name'], axis=0, inplace=True)
                # Если ДФ не пустой
                if len(df) < len_df:
                    print(file)
                    idx = int(file.name.split('.')[0])
                    data = self.get_vacancies_page(idx)
                    found = data.get('found', 0)
                    if not found:
                        file.unlink()
                        print(f'Вакансий для файла {file.name} нет!!!')
                    else:
                        idxs.add(idx)
        return idxs

    def incomplete_df_vacancies_from_csv(self):
        print('Формирование списка работодателей с пропущенными полями...')
        idxs = set()
        files_path = Path(self.parsed_employers)
        pbar = tqdm(total=sum(1 for _ in files_path.glob('[01]*.csv')))
        for file in files_path.glob('[01]*.csv'):
            # если файл не пустой
            if file.stat().st_size > 200:
                df = pd.read_csv(file, sep=';')
                len_df = len(df)
                df.dropna(subset=self.keys_not_item, axis=0, inplace=True)
                # Если ДФ не пустой
                if len_df > len(df):
                    idx = int(file.name.split('.')[0])
                    data = self.get_vacancies_page(idx)
                    found = data.get('found', 0)
                    if found:
                        idxs.add(idx)
                    else:
                        # file.unlink()
                        # print(f'Вакансий для файла {file.name} нет!!!')
                        pass
            pbar.update(1)
        pbar.close()
        return idxs

    def incomplete_df_vacancies_from_csv_multi(self, num_threads=7):
        """
        Многопоточная загрузка Файлов с вакансиями
        :param num_threads: Количество потоков
        :return: None
        """
        self.idxs_employers = set()
        print('Формирование списка работодателей с пропущенными полями...')
        # Вызов функции управления потоками и передача в неё параметров
        self.multithreaded_processing(num_threads,
                                      (self.csv_queue_files,),
                                      self.csv_file_reader,
                                      self.csv_file_writer)
        return self.idxs_employers

    def csv_queue_files(self):
        files_path = Path(self.parsed_employers)
        for file in files_path.glob('[01]*.csv'):
            # если файл не пустой
            if file.stat().st_size > 200:
                # создаем и заполняем очередь именами файлов
                self.files_queue.put(file)
        self.horizontal_bar = tqdm(total=self.files_queue.qsize())

    def csv_file_writer(self):
        """
        ПИСАТЕЛЬ
        :return: None
        """
        # self.idxs_employers = pd.DataFrame()
        while True:
            # ожидаем получение данных
            if self.data_queue.empty():
                # !!!!! проверяем: живы ли потоки читателей file_reader() !!!!!
                if self.event_reader.is_set():
                    self.horizontal_bar.close()
                    # если очередь пуста и все читатели завершили работу - ТО:
                    print('Все файлы обработаны.')
                    print_time(self.rq_time)
                    # конец работы - завершаем цикл
                    break
            else:
                # как только поступили данные извлекаем их и записываем
                idx = self.data_queue.get()
                self.idxs_employers.add(idx)
                # if not len(self.idxs_employers):
                #     self.idxs_employers = idx
                # else:
                #     self.idxs_employers = pd.concat([self.idxs_employers, idx])

    def csv_file_reader(self, idx_thread):
        """
        ЧИТАТЕЛЬ: здесь читаем и обрабатываем данные файлов
        :param idx_thread: номер потока
        :return: None
        """
        # здесь читаем и обрабатываем данные файлов
        while True:
            # Проверяем, есть ли файлы в очереди
            if self.files_queue.empty():
                # выходим из цикла
                break
            self.horizontal_bar.update(1)
            # Получаем имя файла из очереди
            file = self.files_queue.get()
            df = pd.read_csv(file, sep=';')
            len_df = len(df)
            df.dropna(subset=self.keys_not_item, axis=0, inplace=True)
            # Если ДФ не пустой
            if len_df > len(df):
                idx = int(file.name.split('.')[0])
                # data = self.get_vacancies_page(idx)
                # found = data.get('found', 0)
                # if found:
                self.data_queue.put(idx)
            # flt = df[self.keys_not_item].isna().all(axis=1)
            # full_nan = df[flt]
            # if len(full_nan):
            #     self.data_queue.put(full_nan)

    def incomplete_vacancies_multi(self, num_threads=7, max_rows=None):
        """
        Многопоточная загрузка вакансий работодателей
        :param num_threads: Количество потоков
        :param max_rows: Количество записей для парсинга
        :return: None
        """
        # Вызов функции управления потоками и передача в неё параметров
        self.multithreaded_processing(num_threads,
                                      (self.make_queue_vacancies, max_rows),
                                      self.vacancies_reader,
                                      self.vacancies_writer)

    def load_incomplete_vacancies_from_csv(self):
        if os.access(self.incomplete_vacancies, os.F_OK):
            self.df_incomplete = pd.read_csv(self.incomplete_vacancies,
                                             sep=';')
            pattern = re.compile('(?<=/)\d*(?=\?)')
            self.df_incomplete['id'] = self.df_incomplete['url'].apply(
                lambda x: re.findall(pattern, x)[0])
            self.df_incomplete['id'] = self.df_incomplete['id'].astype(int)
            self.df_incomplete.drop_duplicates(subset=['id'], keep='last',
                                               ignore_index=True,
                                               inplace=True)
        else:
            self.df_incomplete = pd.DataFrame(
                columns=self.dict_keys + ('snippet', 'url'))
        self.rows_in_incomplete = len(self.df_incomplete)

    def make_queue_vacancies(self, max_rows=None):
        """
        Формирование очереди вакансий для парсинга
        :param max_rows: Количество записей для парсинга
        :return:
        """
        # чтение файла с вакансиями
        self.load_df_vacancies_from_csv()
        flt = self.df_vacancies[self.keys_not_item].isna().all(axis=1)
        full_nan = self.df_vacancies[flt].sort_values('name', ascending=False)
        print('Кол-во неполных вакансий:', len(full_nan))

        # чтение файла с неполными вакансиями
        self.load_incomplete_vacancies_from_csv()
        idxs = set(self.df_incomplete['id'].to_list())
        print(f'Кол-во дозагруженных вакансий: {len(idxs)}')

        # чтение файла с несуществующими вакансиями
        self.read_errors_vacancies_log()
        print(f'Кол-во несуществующих вакансий: {len(self.errors_vacancies)}')
        idxs = idxs | self.errors_vacancies

        if not max_rows:
            max_rows = len(full_nan)
        random_index = random.choices(population=full_nan.index, k=max_rows)
        for idx in random_index:
            row = full_nan.loc[idx, :]
            if row['id'] not in idxs:
                self.files_queue.put(row)
        print(f'Количество вакансий для обработки: {self.files_queue.qsize()}')

    def vacancies_reader(self, idx_thread):
        """
        ЧИТАТЕЛЬ: Получение вакансий
        :param idx_thread: номер потока
        :return: None
        """
        while True:
            # Проверяем, есть ли данные в очереди
            if self.files_queue.empty():
                print(f'Поток {idx_thread} завершен.')
                # выходим из цикла
                break
            # если установлена задержка - установим
            if self.delay:
                time.sleep(self.get_time_sleep())
            # Получаем индекс и наименование вакансии из очереди
            row = self.files_queue.get()
            print(f'Вакансия: id={row.id} {row.url}')
            enriched = self.download_page(row.url)
            if enriched.get('errors', None) is None:
                # Помещаем данные в выходную очередь
                self.data_queue.put((row, enriched))
                self.new_vacancy_rows += 1
            else:
                if enriched.get('description', None) is None:
                    # словили ошибку про требование ввода капчи --> ставим
                    # глобальный флаг, чтобы больше не читать вакансии
                    # очистка очереди и выход, т.к. не имеет смысла гонять цикл
                    print('Словили капчу!!!')
                    self.errors_captcha_required = True
                    self.files_queue.queue.clear()
                    break
                else:
                    # получили ошибку description": "Not Found"
                    print(f'Ошибка: нет такой вакансии: {row.url}')
                    self.errors_vacancies.add(row.id)
                    # запишем в лог несуществующей вакансии
                    with open(self.errors_vacancies_log, 'a+') as fw:
                        fw.write(f'{row.id};{row.url}\n')

    def vacancies_writer(self):
        """
        ПИСАТЕЛЬ
        :return: None
        """
        while True:
            # ожидаем получение данных
            if self.data_queue.empty():
                # !!!!! проверяем: живы ли потоки читателей _reader() !!!!!
                if self.event_reader.is_set():
                    if self.new_vacancy_rows:
                        self.save_incomplete_vacancies_to_csv()
                    # если очередь пуста и все читатели завершили работу - ТО:
                    print_time(self.rq_time)
                    # конец работы - завершаем цикл
                    break
            else:
                # как только поступили данные извлекаем их и записываем в ДФ
                row, enriched = self.data_queue.get()
                for key in self.keys_not_item:
                    row[key] = enriched.get(key)
                    if key == 'description' and row['description']:
                        description = row['description']
                        row['description'] = ' '.join(description.split())
                # добавляем данные в итоговый ДФ
                self.df_incomplete.loc[len(self.df_incomplete)] = row
                # Если вакансий добавлено более 200 записей сохраним ДФ
                if len(self.df_incomplete) - self.rows_in_incomplete > 200:
                    self.save_incomplete_vacancies_to_csv()

    def save_incomplete_vacancies_to_csv(self):
        print(f'Получено вакансий {self.new_vacancy_rows}')
        print('Сохраняю вакансии в .csv')
        # переименуем старый файл если он есть
        if os.access(self.incomplete_vacancies, os.F_OK):
            old_vacancies_files = os.path.join(self.path_to_save_files,
                                               'old_vacancies')
            if not os.path.exists(old_vacancies_files):
                os.makedirs(old_vacancies_files)
            last_time = os.path.getmtime(self.incomplete_vacancies)
            last_time = datetime.utcfromtimestamp(last_time)
            last_time = last_time.strftime("%Y-%m-%dT%H_%M_%S")
            old_file = os.path.basename(self.incomplete_vacancies)
            old_file = old_file.replace('.csv', f'_{last_time}.csv')
            old_file = os.path.join(old_vacancies_files, old_file)
            # если файл не был сохранен ранее - сохраним его
            if not os.access(old_file, os.F_OK):
                os.rename(self.incomplete_vacancies, old_file)
        self.df_incomplete.to_csv(self.incomplete_vacancies, sep=';',
                                  index=False)
        self.rows_in_incomplete = len(self.df_incomplete)
        print(f'Сохранено {self.rows_in_incomplete} вакансий в .csv')

    def read_errors_vacancies_log(self):
        """
        Чтение из лог-файла идентификаторов несуществующих вакансий
        :return: None
        """
        if os.access(self.errors_vacancies_log, os.F_OK):
            tmp = pd.read_csv(self.errors_vacancies_log, header=None, sep=';')
            tmp.columns = ['id', 'url']
            self.errors_vacancies = set(tmp['id'].values)


if __name__ == "__main__":
    hh_obj = ParsingHH()

    # hh_obj.make_all_employers()

    # hh_obj.get_vacancies_pages(4167)
    # hh_obj.employers_vacancies_multi(num_threads=1, idxs_employers=[3592])

    # res = hh_obj.make_patterns(is_IT=True)
    # print(res)

    # hh_obj.getting_numbers_vacancy_resume(kind='resume')
    # hh_obj.getting_numbers_vacancy_resume(kind='resume', search='specialization')
    # hh_obj.getting_numbers_vacancy_resume(kind='resume', search='professional_role')

    # hh_obj.getting_number_resumes()

    # res = hh_obj.get_selected_resume_info()
    # print(res.get('found', 0))

    result = hh_obj.make_dict_specializations(return_category=True)
    result = {key: val[0] for key, val in result.items()}
    print(*result.items(), sep='\n')
    df = pd.DataFrame(data=result.items(), columns=['id', 'specializations'])
    df.to_csv('specializations.csv', index=False, sep=';', encoding='cp1251')
    #
    # hh_obj.make_dict_professional_roles()
    # print(*hh_obj.category_prof_roles.items(), sep='\n')

    # temp = hh_obj.files_from_job_monitoring_to_df()
    # print(temp)
    # print(temp.info())

    # # сбор данных по вакансиям и резюме из файлов
    # temp = hh_obj.join_vacancy_resume(search='specialization', only_IT=False)
    # temp.to_excel(os.path.join(hh_obj.path_file, 'text.xlsx'), index=False)
    # # print(temp)

    # объединение файлов с вакансиями
    # hh_obj.make_df_vacancies_from_csv()

    # чтение файла с вакансиями
    # hh_obj.load_df_vacancies_from_csv()
    # print(hh_obj.df_vacancies.info())

    # idxs_employers = hh_obj.make_bad_df_vacancies_from_csv()
    # print(idxs_employers)
    # # hh_obj.employers_vacancies_multi(num_threads=5,
    # #                                  idxs_employers=idxs_employers)
    # for idx in idxs_employers:
    #     print(idx)
    #     data = hh_obj.get_vacancies_page(idx)
    #     found = data.get("found", 0)
    #     print(f'id={idx} Вакансий={data.get("found", 0)}')

    # temp = hh_obj.doubtful_df_vacancies_from_csv()
    # print(len(temp))

    # df = hh_obj.load_employer_vacancies(17552)
    # print(df.info())
    # print('54598309' in df['id'].values)
    # print(set(df['id'].to_list()))

    # hh_obj.checking_existing_vacancies = True
    # hh_obj.employers_vacancies_multi(num_threads=1, idxs_employers=[17552])

    # file = Path(hh_obj.parsed_employers).joinpath('00721052.csv')
    # tmp = pd.read_csv(file, sep=';')
    #
    # filter_missing = tmp[hh_obj.keys_not_item].isna().all(axis=1)
    # vacancies_exist = set(tmp[filter_missing]['id'].to_list())
    # # vacancies_exist = set(vacancies_exist['id'].to_list())
    # print(vacancies_exist)

    #     tmp.loc[row.Index, 'id'] = row.Index * 100
    # print(tmp)

    # hh_obj.make_queue_vacancies()
    # df = hh_obj.df_vacancies

    # hh_obj.load_incomplete_vacancies_from_csv()
    # df = hh_obj.df_incomplete
    # df['id_in_url'] = df.apply(
    #     lambda row: pd.isna(row.url) or str(row.id) in row.url, axis=1)
    # df['nid'] = df.url.apply(
    #     lambda x: int(re.findall('(?<=/)\d*(?=\?)', x)[0]))
    # print(df[df.id_in_url])
    # print(df[~df.id_in_url][['id', 'url', 'nid']])
    # print(df.info())

    # idxs = hh_obj.incomplete_df_vacancies_from_csv()
    # idxs = hh_obj.incomplete_df_vacancies_from_csv_multi()

    # print(sorted(idxs)[:7])
    # print(len(idxs))
    # print(len(hh_obj.idxs_employers))
    # dframe = pd.DataFrame(idxs, columns=['file_name'])
    # dframe.to_csv('incomplete_vacancies', index=False)
    # hh_obj.idxs_employers.to_csv('incomplete_vacancies.csv', sep=';',
    #                              index=False)

    # hh_obj.load_missing_vacancy_fields = True
    # hh_obj.employers_vacancies_multi(num_threads=1, idxs_employers=[721052])

    # files = [file for file in Path(hh_obj.parsed_employers).glob('*.csv')]
    # print(len(files))
    # print(files[:5])
    # print(files[-5:])
