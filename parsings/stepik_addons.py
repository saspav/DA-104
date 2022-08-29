"""
Поиск по CSS селекторам (справочное руководство)
https://docs-python.ru/packages/paket-beautifulsoup4-python/css-selektory/
"""

# импорт библиотек
import os
import re
import pickle
import pandas as pd
import requests
from urllib.parse import unquote
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import \
    visibility_of_element_located as elem_vis
from selenium.webdriver.support.expected_conditions import \
    presence_of_all_elements_located as elem_loc
from selenium.webdriver.support.expected_conditions import \
    element_to_be_clickable as elem_click
from selenium.common.exceptions import NoSuchElementException
from pathlib import Path
from time import sleep
from datetime import datetime
from configparser import ConfigParser
from math import log, sin
from tqdm import tqdm


class MakeTask:
    """
    Класс решения задания и загрузки ответа на страницу урока
    """

    def __init__(self, base_dir=None):
        """
        Создание экземпляра драйвера браузера и авторизация на сайте степика
        :param base_dir: каталог, где лежат куки авторизации на степике
        :return: None
        """
        # базовый каталог
        if base_dir:
            self.BASE_DIR = base_dir
        else:
            self.BASE_DIR = r'D:\python-txt\selenium_python'
        # если такого каталога не существует - создадим
        if not os.path.isdir(self.BASE_DIR):
            os.makedirs(self.BASE_DIR)
        # Инициализируем драйвер
        self.browser = webdriver.Chrome()
        # результат решения задачи
        self.lesson_url = ''
        self.task_url = ''
        self.result = ''

    def login_stepik(self, base_dir=None):
        """
        Ручная Авторизация на сайте степика и сохранение этого в куки
        :param base_dir: каталог, где лежат куки авторизации на степике
        :return: None
        """
        if not base_dir:
            base_dir = self.BASE_DIR
        self.browser.get('https://stepik.org/catalog?auth=login')
        # WebDriverWait указывает время ожидания в секундах
        wait = WebDriverWait(self.browser, 99)
        # css-селектор элемента который мы ждем
        items_selector = '#ember15'
        # Здесь мы ждем появления элемента item_selector,
        # если в течение 99 секунд он не появится, произойдет исключение
        wait.until(elem_vis((By.CSS_SELECTOR, items_selector)))
        # сохранение Куки в файл
        with open(os.path.join(base_dir, 'user_chrome.pkl'), 'wb') as f:
            pickle.dump(self.browser.get_cookies(), f)
        self.browser.quit()

    def auth_stepik(self, base_dir=None):
        """
        Авторизация на сайте степика
        :param base_dir: каталог, где лежат куки авторизации на степике
        :return: None
        """
        if not base_dir:
            base_dir = self.BASE_DIR
        # вход на первую страницу
        self.browser.get('https://stepik.org/catalog')
        # читаем сохраненные куки
        cookies = pickle.load(open(os.path.join(base_dir, 'user_chrome.pkl'),
                                   'rb'))
        # добавляем куки в браузер
        for cookie in cookies:
            self.browser.add_cookie(cookie)
        self.browser.refresh()

    def get_answer(self, task_url=None):
        """
        Получаем решение
        :param task_url: ссылка на задание
        :return: результат
        """
        if not task_url:
            task_url = self.task_url
        self.result = self.browser.get(task_url).text
        return self.result

    def wait_spinner(self):
        """
        Ждём, пока спиннер пропадёт
        :return: None
        """
        wait = WebDriverWait(self.browser, 9)
        for class_name in ('loader__spinner', 'stepik-loader'):
            wait.until_not(elem_loc((By.CLASS_NAME, class_name)))
        for txt in ('подождите, не покидайте страницу',
                    'С этого шага можно безопасно уходить'):
            wait.until_not(
                elem_loc((By.XPATH, f'//span[contains(text(),{txt})]')))

    def send_answer(self, lesson_url=None, result=None):
        """
        Вставка решения на страницу урока
        :param lesson_url: линк на урок
        :param result: текст решения
        :return: None
        """
        if not lesson_url:
            lesson_url = self.lesson_url
        if not result:
            result = self.result

        wait = WebDriverWait(self.browser, 9)
        # идем на страницу урока
        self.browser.get(lesson_url)
        self.wait_spinner()
        # если есть решение - обновим
        if self.browser.find_elements(By.CLASS_NAME, 'again-btn'):
            self.browser.find_element(By.CLASS_NAME, 'again-btn').click()
            if self.browser.find_elements(By.CLASS_NAME,
                                          'modal-popup__container'):
                # подтвердим, что хотим решить снова
                self.browser.find_element_by_xpath(
                    '//button[text()="OK"]').click()
                # очистим поле "решение"
                wait.until(elem_click((By.TAG_NAME, "textarea"))).clear()
        # вставка ответа
        wait.until(elem_click((By.TAG_NAME, "textarea"))).send_keys(result)
        # отправка решения
        self.browser.find_element(By.CLASS_NAME, 'submit-submission').click()
        self.wait_spinner()
        if self.browser.find_elements(By.CLASS_NAME, 'attempt-message_wrong'):
            print('Неверное решение!!!')
            # что-то пошло не так подержим браузер 30 сек
            __import__('time').sleep(30)
        else:
            print('Решено верно!!!')

    def solution(self, task_url=None, lesson_url=None):
        if not task_url:
            task_url = self.task_url
        if not lesson_url:
            lesson_url = self.lesson_url
        try:
            # получаем решение
            answer = self.get_answer(task_url)
            print(answer)
            # отправляем решение
            self.auth_stepik()
            self.send_answer(lesson_url, answer)
        finally:
            self.browser.quit()

    @staticmethod
    def calc(x):
        return str(log(abs(12 * sin(int(x)))))


class ReadINI:
    """
    Класс чтения параметров из ini-файла
    """

    def __init__(self, base_dir=None, ini_file=None, course_name=None):
        """
        Чтение данных о курсе из конфигурационного файла 'stepik.ini'
        :param base_dir: базовый каталог
        :param course_name: кодовое имя курса
        :return: None
        """
        if ini_file:
            self.ini_file = ini_file
        else:
            self.ini_file = 'stepik.ini'
        if not course_name:
            raise ValueError(f'Ошибка: не задано имя курса')
        # базовый каталог
        if base_dir:
            self.BASE_DIR = base_dir
        else:
            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # файл с настройками ищем в текущем каталоге
        full_path = os.path.join(self.BASE_DIR, self.ini_file)
        if os.path.isfile(full_path):
            # читаем файл с настройками
            config = ConfigParser()
            config.read(full_path)
            # читаем значения из соответствующей секции
            self.title = config.get(course_name, 'title')
            self.url_course = config.get(course_name, 'url_course')
            # ссылка на первую страницу курса
            self.sql_url = config.get(course_name, 'sql_url')
            self.lessons_page = config.get(course_name, 'lessons_page')

            # имя файла с обработанными шагами курса
            self.processed_steps = self.lessons_page.replace('.html',
                                                             '_processed.csv')
            if not os.path.isdir(os.path.join(self.BASE_DIR, 'LOGS')):
                os.makedirs(os.path.join(self.BASE_DIR, 'LOGS'))
        else:
            raise FileNotFoundError(f'Ошибка: не найден файл {full_path}')


class MakeHTML:
    """
    Класс для создания html курса путем скачивания со степика
    """

    def __init__(self, base_dir=None):
        """
        :param base_dir: базовый каталог
        """
        self.title = ''  # название курса
        self.url_course = ''  # ссылка на курс
        self.html_text = ''  # содержимое html (файл или страница)
        # суп из шаблона страницы, в которую будем добавлять шаги курса
        self.html_soup = BeautifulSoup(self.html_text, 'html.parser')
        self.step_url = None  # ссылка на шаг
        self.step_id = None  # идентификатор шага
        self.browser = None  # экземпляр браузера
        self.page_source = None  # содержимое страницы браузера
        self.now_step = None  # содержимое текущего шага (заголовок + текст)
        self.now_attempt = None  # решение текущего шага
        # суп из страницы браузера текущего шага
        self.step_soup = BeautifulSoup(self.html_text, 'html.parser')
        # код строки-разделителя
        self.divider_string = '<hr class="modern-lesson-divider">\n'
        # базовый каталог
        if base_dir:
            self.BASE_DIR = base_dir
        else:
            self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        # каталог куда будут складываться создаваемые html-файлы
        self.HTML_DIR = os.path.join(self.BASE_DIR, 'HTML_DIR')
        # если такого каталога не существует - создадим
        if not os.path.isdir(self.HTML_DIR):
            os.makedirs(self.HTML_DIR)
        # если такого каталога для логов не существует - создадим
        if not os.path.isdir(os.path.join(self.BASE_DIR, 'LOGS')):
            os.makedirs(os.path.join(self.BASE_DIR, 'LOGS'))
        # файл с обработанными страницами
        self.processed_steps = None
        # количество строк в поле "решение"
        self.attempt_lines = -1
        # датафрейм с обработанными страницами
        self.processed_urls = pd.DataFrame(columns=['url', 'id', 'lines'])
        # каталог с загруженными стилями и картинками
        self.FILE_DIR = 'stepik_files'
        self.url_file = None  # адрес (ссылка) картинки
        # теги, по которым выбирать ссылки на картинки
        self.tags = dict(img=['src'], script=['src'], source=['src'])
        self.links = set()  # множество ссылок на картинки
        self.link_file = dict()  # словарь: ссылка - имя файла
        self.picture_file = None  # имя загруженного файла с картинкой
        self.picture_full = None  # полное имя загруженного файла

    def clear_step(self):
        """
        Очистка переменных с содержимым шага и решения
        :return: None
        """
        # получение идентификатора шага из self.step_url шага (только цифры)
        self.step_id = '_'.join(re.findall(r'\d+',
                                           self.step_url.split('?')[0]))
        html_txt_step = f'<s_work id="work_{self.step_id}">\n</s_work>'
        html_txt_shot = f'<s_attempt id="shot_{self.step_id}">\n</s_attempt>'
        self.now_step = BeautifulSoup(html_txt_step, 'html.parser')
        self.now_attempt = BeautifulSoup(html_txt_shot, 'html.parser')

    def step_append_divider(self):
        """
        Добавление строки-разделителя в содержимое шага
        :return: None
        """
        self.now_step.s_work.append(BeautifulSoup(self.divider_string,
                                                  'html.parser'))

    def soup_page(self, html_text=''):
        """
        Парсинг html текста страницы шаблона курса
        :return: None
        """
        if not html_text:
            html_text = self.html_text
        self.html_soup = BeautifulSoup(html_text, 'html.parser')

    def add_page_title(self):
        """
        Добавление наименования и ссылки на курс
        :return: None
        """
        if self.title:
            new_soup_tag = self.html_soup.new_tag('title')
            new_soup_tag.string = self.title
            self.html_soup.head.append(new_soup_tag)
        if self.url_course:
            new_soup_tag = self.html_soup.new_tag("a", href=self.url_course)
            new_soup_tag.string = self.title
            self.html_soup.tag_title.append(new_soup_tag)
            self.html_soup.tag_title.a.string.wrap(self.html_soup.new_tag("u"))
            self.html_soup.tag_title.a.wrap(self.html_soup.new_tag("h2"))

    def add_body_child(self, html_text):
        """
        Добавление элемента на страницу курса в тег "tag_job"
        :param html_text: html с содержанием и решением шага
        :return: None
        """
        self.html_soup.body.tag_job.append(html_text)

    def save_html_file(self, name_page, html_text=None, dir_page='',
                       sub_dir=''):
        """
        Сохранение html в файл
        :param name_page: имя файла
        :param html_text: html текст
        :param dir_page: каталог
        :param sub_dir: подкаталог
        :return: None
        """
        if not html_text:
            html_text = self.html_soup
        if not dir_page:
            dir_page = self.HTML_DIR
        # если такого каталога не существует - создадим
        if not os.path.isdir(dir_page):
            os.makedirs(dir_page)
        # если такого каталога не существует - создадим
        if sub_dir and not os.path.isdir(os.path.join(dir_page, sub_dir)):
            os.makedirs(os.path.join(dir_page, sub_dir))
        fileHTML = open(os.path.join(dir_page, sub_dir, name_page), 'w',
                        encoding='UTF-8')
        fileHTML.write(str(html_text))
        fileHTML.close()

    def read_html_file(self, name_page, dir_page='', sub_dir=''):
        """
        Чтение html из файла
        :param name_page: имя файла
        :param dir_page: каталог
        :param sub_dir: подкаталог
        :return: содержимое html - файла
        """
        if not dir_page:
            dir_page = self.HTML_DIR
        fileHTML = open(os.path.join(dir_page, sub_dir, name_page),
                        encoding='UTF-8')
        html_text = fileHTML.read()
        fileHTML.close()
        return html_text

    def read_processed_steps(self):
        name_file = os.path.join(self.BASE_DIR, 'LOGS', self.processed_steps)
        if os.path.exists(name_file):
            self.processed_urls = pd.read_csv(name_file)

    def add_processed_steps(self, url=None):
        if not url:
            url = self.step_url
        self.processed_urls.loc[len(self.processed_urls)] = [url, self.step_id,
                                                             self.attempt_lines]

    def write_processed_steps(self):
        name_file = os.path.join(self.BASE_DIR, 'LOGS', self.processed_steps)
        self.processed_urls.to_csv(name_file, index=False)

    def make_browser(self):
        """
        Создание экземпляра драйвера браузера и авторизация на сайте степика
        :return: None
        """
        # Инициализируем драйвер
        option = webdriver.FirefoxOptions()
        option.set_preference('dom.webdriver.enabled', False)
        self.browser = webdriver.Firefox(options=option)
        # вход на первую страницу
        self.browser.get('https://stepik.org/catalog')
        # читаем сохраненные куки
        cookies = pickle.load(open(os.path.join(self.BASE_DIR, 'stepik.pkl'),
                                   'rb'))
        # добавляем куки в браузер
        for cookie in cookies:
            self.browser.add_cookie(cookie)

    def get_page(self, page_url, wait_sidebar=True):
        """
        Чтение страницы по ссылке
        :param page_url: адрес страницы
        :param wait_sidebar: ожидать полной загрузки левого сайдбара
        :return: сайдбар (или элемент "кнопка следуюший шаг"),
        содержимое страницы
        """
        self.browser.get(page_url)

        wait = WebDriverWait(self.browser, 30)

        # Ждём, пока спиннер пропадёт
        wait.until_not(elem_loc((By.CLASS_NAME, 'loader__spinner')))
        wait.until_not(elem_loc((By.CLASS_NAME, 'stepik-loader')))
        if wait_sidebar:
            sleep(9)
            # Ждём, пока все данные сбоку загрузятся
            wait.until_not(elem_loc((By.CLASS_NAME,
                                     'lesson-sidebar__lesson-name-placeholder')))
            # Достаём сайдбар
            left_bar = self.browser.find_element(By.CLASS_NAME,
                                                 'lesson-sidebar__content')
            # print(sidebar.text)
        else:
            # ждем появление кнопки "следующий шаг"
            left_bar = wait.until(elem_click((By.CLASS_NAME,
                                              'lesson__next-btn')))
        self.page_source = self.browser.page_source
        return left_bar, self.page_source

    def step_make_soup(self, html_doc=''):
        """
        Приготовление супа из прочитанной страницы
        :param html_doc:
        :return: None
        """
        if not html_doc:
            html_doc = self.page_source
        self.step_soup = BeautifulSoup(html_doc, 'lxml')

    def add_lesson_name(self, lesson_name):
        """
        Добавление названия урока из сайдбара на страницу шага
        :param lesson_name: Наименование урока
        :return: None
        """
        # добавление строки-разделителя
        self.step_append_divider()
        # добавление названия урока из сайдбара
        soup_tag_h3 = self.now_step.new_tag("h3")
        soup_tag_h3.string = lesson_name
        self.now_step.s_work.append(soup_tag_h3)
        # добавление ссылки на урок
        soup_tag_a = self.now_step.new_tag("a", href=self.step_url)
        soup_tag_a.string = self.step_url
        # заверенем в тэг "i" - курсив
        soup_tag_a.string.wrap(self.now_step.new_tag("i"))
        self.now_step.s_work.append(soup_tag_a)
        self.step_append_divider()

    def add_step_top(self):
        """
        Формирование блока-заголовка шага с названием раздела и шага
        :return: None
        """
        # получение номера шага
        num_step = re.search(r'(?<=step/)\d+', self.step_url)
        # получение наименования шага
        step_name = self.step_soup.select('.top-tools__lesson-name')
        # формируем тег "h4" с названием урока
        soup_tag_h4 = self.now_step.new_tag("h4")
        soup_tag_h4.string = f'  {str(step_name[0].string).strip()}. '
        # формируем тег "a" со ссылкой на шаг
        soup_tag_a = self.now_step.new_tag("a", href=self.step_url)
        soup_tag_a.string = f'Шаг: {num_step[0]}.' if num_step else ''
        # добавление строки-разделителя
        self.step_append_divider()
        # добавляем тег "h4" с названием урока
        self.now_step.s_work.append(soup_tag_h4)
        # внутрь тег "h4" добавляем тэг "a" со ссылкой на шаг
        self.now_step.s_work.h4.append(soup_tag_a)
        # текст у ссылки завернем в тэг "u"
        self.now_step.s_work.h4.a.string.wrap(self.now_step.new_tag("u"))
        # добавление строки-разделителя
        self.step_append_divider()

    def add_lesson_content(self):
        """
        Формирование содержимого шага
        :return: None
        """
        # Достаем содержимое шага (содержание урока)
        content = self.step_soup.select('.html-content')[0]
        # Если это содержание урока заменим тег h2 на h3, если нет - на h4
        if content.h2 and content.h2.string.lower()[1:] == 'одержание урока':
            tag = content.h2
            tag.name = 'h3'
        else:
            if content.h2:
                tag = content.h2
                tag.name = 'h4'
        # добавление на страницу содержимое шага
        self.now_step.s_work.append(content)
        # формирование множества ссылок из содержимого шага
        self.links_from_content(content)

    def links_from_content(self, content):
        """
        формирование множества ссылок из содержимого шага
        :param content: содержимое шага
        :return: None
        """
        # Перебираем теги с картинками
        for tag1, tags2 in self.tags.items():
            for link_url in content.select(tag1):
                for tag2 in tags2:
                    # получаем ссылку, соответствующую тегу
                    url = link_url.get(tag2)
                    if url and (url.lower().startswith('http')
                                or tag2 in ['playlist-url']):
                        self.links.add(url)
                        # print(f'url: {url}')

    def find_step_attempt(self):
        """
        Поиск на странице кода с решением
        :return: bool True - найдено решение, False - нет
        """
        found_step_attempt = True
        try:
            _ = self.browser.find_element(By.CLASS_NAME, 'attempt')
        except NoSuchElementException:
            found_step_attempt = False
        return found_step_attempt

    def make_step_attempt(self):
        """
        Формирование нового блока с кодом решения
        :return: None
        """
        # Если это блок с текстом решения - удалим лишние строки
        if self.step_soup.select('.CodeMirror-code'):
            attempt = self.step_soup.select('.CodeMirror-code')[0]
            # print(attempt.contents)
            # print(len(attempt.contents))
            indexes = []
            for child in attempt.children:
                # print(child)
                if re.search('<span cm-text="">', str(child)):
                    indexes.append(True)
                else:
                    indexes.append(False)
            # удаление пустых строк в решении
            if len(indexes):
                for i in range(len(indexes))[::-1]:
                    if indexes[i]:
                        indexes.pop(i)
                    else:
                        break
            # формирование блока с кодом решения
            txt = '<div class="CodeMirror-code" role="presentation" style="">'
            for i in range(len(indexes)):
                codes = self.step_soup.select(
                    f'.CodeMirror-code > div:nth-child({i + 1})')
                txt += str(codes[0])
            txt += r'</div>'
            attempt_soup = BeautifulSoup(txt, 'html.parser')
            # print(attempt_soup)
            self.now_attempt.s_attempt.append(attempt_soup)
            self.attempt_lines = len(indexes)
        else:
            attempt = self.step_soup.select('.attempt')[0]
            self.now_attempt.s_attempt.append(attempt)
            self.attempt_lines = -1
        # добавление пустой строки после решения
        subject = self.now_attempt.new_tag('h5')
        subject.string = '\n'
        self.now_attempt.s_attempt.append(subject)

    def add_step_attempt(self):
        """
        Добавление блока с кодом решения на страницу с шагом
        :return: None
        """
        att_top = """
        <div class="attempt-main ember-view">
            <div class="attempt-wrapper correct sql">
            <!---->
            <h3 class="quiz__typename">Решение:</h3>
            <section class="attempt-wrapper__body">
                <div class="attempt__inner">
                    <div class="attempt">
                    <div class="quiz-plugin ember-view">
                    <div class="show-plugin">
                    <div class="CodeMirror cm-s-default CodeMirror-wrap">
                    <div class="code-editor ember-view">
                    <div class="CodeMirror-scroll">
                        <div class="CodeMirror-sizer" style="margin-left: 30px; 
                            margin-bottom: -22px; border-right-width: 8px; 
                            min-height: 44px; padding-right: 0px; 
                            padding-bottom: 0px;">
                            <div style="position: relative; top: 0px;">
                                <div class="CodeMirror-lines">
        """
        att_footer = """
                                </div> 
                            </div>
                        </div>
                        <div style="position: absolute; height: 8px; width: 1px; border-bottom: 0px solid transparent; top: 176px;"></div>
                        <div class="CodeMirror-gutters" style="height: 184px; left: 0px;">
                            <div class="CodeMirror-gutter CodeMirror-linenumbers" style="width: 29px;"></div>
                        </div>
                    </div> </div> </div> </div> </div> </div> 
                </div>
                <div class="attempt__actions"> </div>
            </section>
            </div>	
        </div>
        """
        self.now_attempt = f'{att_top}\n{str(self.now_attempt)}\n{att_footer}'
        self.now_attempt = BeautifulSoup(self.now_attempt, 'html.parser')
        self.now_step.s_work.append(self.now_attempt)

    def make_lessons_links(self, name_html):
        """
        Формирование ссылок на картинки из сохраненного файла
        :param name_html:
        :return: None
        """
        # чтение файла с уроками
        self.page_source = self.read_html_file(name_html)
        # готовим суп из страницы
        self.step_make_soup()
        # получаем ссылки из всего супа
        self.links_from_content(self.step_soup)

    def make_full_name(self, url):
        """
        Формирование полного имени файла
        :param url: None
        :return:
        """
        # Формирование полного имени файла
        self.picture_full = os.path.join(self.HTML_DIR, self.FILE_DIR,
                                         self.picture_file)
        # добавление в словарь ссылки и имени файла с картинкой
        self.link_file[url] = f'{self.FILE_DIR}/{self.picture_file}'

    def check_size_files(self, url, set_file_size=True):
        """
        Проверка соответствия размеров файла по ссылке и файла на диске
        если размеры файлов разные - сохраним его под уникальным именем
        :param url: адрес картинки
        :param set_file_size: bool - True=к имени файла добавить размер
        :return: None
        """
        url_file_size = self.url_file.headers.get('Content-Length', 0)
        now_file_size = Path(self.picture_full).stat().st_size
        # если размеры файлов разные
        if url_file_size and int(url_file_size) != now_file_size:
            # попробуем к имени файла добавить размер или текущий time_stamp
            if set_file_size:
                self.picture_file = f'{url_file_size}_{self.picture_file}'
            else:
                time_stamp = datetime.now().strftime('%y%m%d%H%M%S%f')
                self.picture_file = f'{time_stamp}_{self.picture_file}'
            # Формирование нового полного имени файла
            self.make_full_name(url)

    def make_file_name(self, url):
        """
        Формирование имени файла
        :param url: адрес картинки
        :return: None
        """
        # Формирование полного имени файла
        self.make_full_name(url)
        # если файл существует - проверим его размер
        if os.path.exists(self.picture_full):
            # если файлы разные - к имени файла добавим размер
            self.check_size_files(url, set_file_size=True)
            if os.path.exists(self.picture_full):
                # если файлы разные - к имени файла добавим текущий time_stamp
                self.check_size_files(url, set_file_size=False)

    def write_save_error(self, url):
        """
        Возникла ошибка при сохранении картинки - пишем её в лог
        :param url: ссылка
        :return: None
        """
        with open(os.path.join(self.BASE_DIR, 'err.txt'), 'a',
                  encoding='cp1251') as err_file:
            err_file.write(f'Ошибка при создании файла '
                           f'{self.picture_full} link={url}\n')

    @staticmethod
    def link_to_url(href, only_gb=True):
        """
        Преобразование ссылки href из html-страницы в url для загрузки
        Здесь это фиктивный метод: что принял - то и вернул
        :param href: ссылка со страницы
        :param only_gb: True -> ссылки только с сайта geekbrains
        :return: url для загрузки
        """
        return href

    @staticmethod
    def get_file_from_url(url_link):
        """
        Загрузка содержимого url_link
        Сделано так, чтобы в наследуемых классах можно переопределить
        этот метод, например для запросов на сайт, требующий авторизации
        :param url_link: ссылка для загрузки
        :return: ответ реквеста
        """
        return requests.get(url_link, allow_redirects=True, stream=True)

    def load_files_from_links(self):
        """
        Загрузка картинок по множеству ссылок
        :return:
        """
        for url in self.links:
            # скачивание файла с картинкой
            correct_url = self.link_to_url(url, only_gb=False)
            self.url_file = self.get_file_from_url(correct_url)
            # достаем заголовки
            head = self.url_file.headers
            content_type = head.get('Content-Type', None)
            # print(content_type)
            file_from_url = url.split('/')[-1]
            if '.' not in file_from_url:
                content_type = content_type.split('/')[-1]
                file_from_url = f'file={file_from_url}.{content_type}'
            # получаем имя файла
            self.picture_file = head.get('Content-Disposition',
                                         file_from_url).split('=')[-1]
            self.picture_file = unquote(self.picture_file)
            self.picture_file = re.sub(r'[ (]', '_', self.picture_file)
            self.picture_file = re.sub(r'UTF-8|[ ")\']', '', self.picture_file)
            # Формирование имени файла
            self.make_file_name(url)
            # Если файла нет - сохраняем в каталог по полному пути
            if not os.path.exists(self.picture_full):
                print(f'Сохраняю файл с типом "{content_type}": {url}')
                try:
                    # Новая часть: начало
                    total_size = int(head.get('content-length', 0))
                    block_size = 1024  # 1 K byte
                    progress_bar = tqdm(total=total_size, unit='iB',
                                        unit_scale=True)
                    # открываем файл для записи, в режиме wb и сохраняем
                    with open(self.picture_full, 'wb') as out_file:
                        for data in self.url_file.iter_content(block_size):
                            progress_bar.update(len(data))
                            out_file.write(data)
                    progress_bar.close()
                    if total_size and progress_bar.n != total_size:
                        # возникла ошибка при сохранении - пишем её в лог
                        self.write_save_error(url)
                    #  Новая часть: конец

                    # # открываем файл для записи, в режиме wb и сохраняем
                    # with open(self.picture_full, 'wb') as out_file:
                    #     out_file.write(self.url_file.content)

                except FileNotFoundError:
                    # возникла ошибка при сохранении картинки - пишем её в лог
                    self.write_save_error(url)
        # сохраним словарь
        self.write_link_file()

    def write_link_file(self):
        """
        сохраняем файл с данными: url - имя файла
        :return: None
        """
        if not os.path.isdir(os.path.join(self.BASE_DIR, 'LOGS')):
            os.makedirs(os.path.join(self.BASE_DIR, 'LOGS'))
        name_file = os.path.join(self.BASE_DIR, 'LOGS',
                                 f'uf_{self.processed_steps}')
        # словарь в датафрейм
        df_links = pd.DataFrame(list(self.link_file.items()),
                                columns=['url', 'file'])
        df_links.to_csv(name_file, index=False, sep=';')

    def read_link_file(self):
        """
        Читаем файл с данными: url - имя файла
        :return: словарь url - имя файла
        """
        name_file = os.path.join(self.BASE_DIR, 'LOGS',
                                 f'uf_{self.processed_steps}')
        if os.path.exists(name_file):
            df_links = pd.read_csv(name_file, sep=';')
        else:
            df_links = pd.DataFrame(columns=['url', 'file'])
        link_file = dict()
        for element in df_links.T.to_dict().values():
            link_file[element['url']] = element['file']
        return link_file

    def replace_link_to_file(self, html_doc=''):
        """
        Замена на сформированной странице ссылок на картинки
        ссылками на локальные файлы
        :param html_doc: html текст
        :return:
        """
        # если html текст не передали - используем сформированный из шагов
        if not html_doc:
            html_doc = self.html_soup
        # преобразуем в текст
        html_doc = str(html_doc)
        # изменение url картинок на локальный путь к файлам
        for link_url, name_file in self.link_file.items():
            html_doc = html_doc.replace(link_url, name_file)
        # опять приготовим суп
        self.html_soup = BeautifulSoup(html_doc, 'html.parser')


if __name__ == "__main__":
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #
    # # stepik_ini = ReadINI(base_dir=BASE_DIR, course_name='SQL_EXT')
    # # print(stepik_ini.title)
    # # print(stepik_ini.url_course)
    # # print(stepik_ini.sql_url)
    # # print(stepik_ini.lessons_page)
    # # print(stepik_ini.processed_steps)
    # # raise ValueError
    #
    # empty_html = 'empty.html'
    # main_page = 'index.html'
    # lessons_page = 'test_steps.html'
    # #
    # # весь курс по шагам
    # lessons = MakeHTML(base_dir=BASE_DIR)
    # lessons.processed_steps = 'test_steps.csv'
    # lessons.read_processed_steps()
    # lessons.html_text = lessons.read_html_file(empty_html,
    #                                            dir_page=lessons.BASE_DIR)
    # lessons.soup_page()
    # lessons.title = 'Интерактивный тренажер по SQL'
    # lessons.url_course = 'https://stepik.org/course/63054/syllabus'
    # lessons.add_page_title()
    # #
    # # print(lessons.processed_urls.url.tolist())
    # # print(lessons.processed_urls.iloc[-1].url)
    #
    # # чтение файла с уроками + готовим суп из страницы
    # # lessons.make_lessons_links(lessons_page)
    # #
    # # print(lessons.links)
    # # загрузка файлов на странице
    # # lessons.load_files_from_links()
    # #
    # # [print(f'{key}: {val}') for key, val in lessons.link_file.items()]
    #
    # # # прочитаем словарь url - имя файла
    # # lessons.link_file = lessons.read_link_file()
    # # # изменение url картинок на локальный путь к файлам
    # # lessons.replace_link_to_file(html_doc=lessons.step_soup)
    # # # сохранение html в файл
    # # lessons.html_soup.prettify()
    # # lessons.save_html_file('test.html')
    #
    # # lessons.soup_page()
    # # lessons.title = 'Интерактивный тренажер по SQL'
    # # lessons.url_course = 'https://stepik.org/course/63054/syllabus'
    # # lessons.add_page_title()
    # #
    #
    # lessons.step_url = 'https://stepik.org/lesson/297508/step/8?unit=279268'
    # save_main_page = True
    # if save_main_page:
    #     lessons.make_browser()
    #     lessons.get_page(lessons.step_url)
    #     # wait_sidebar = False
    #     # sidebar, _ = lessons.get_page(lessons.step_url,
    #     #                               wait_sidebar=wait_sidebar)
    #     # sidebar.click()
    #     # sleep(3)
    #     lessons.get_page(lessons.step_url, wait_sidebar=False)
    #     sleep(3)
    #
    #     # сохраним для опытов эту страницу
    #     lessons.save_html_file(lessons_page, html_text=lessons.page_source)
    # else:
    #     # чтение файла с уроками
    #     lessons.page_source = lessons.read_html_file(lessons_page)
    #
    # # инициализация урока
    # lessons.clear_step()
    # # готовим суп из страницы
    # lessons.step_make_soup()
    # # # формирование блока с названием раздела и шага
    # # lessons.add_step_top()
    # # Содержание урока
    # lessons.add_lesson_content()
    # # формирование нового блока с кодом решения
    # lessons.make_step_attempt()
    # # lessons.now_attempt = lessons.step_soup.select('.attempt')
    #
    # # сборка урока в кучу и добавление его в тело общего html
    # lessons.add_step_attempt()
    # # добавление урока в html
    # lessons.add_body_child(lessons.now_step)
    # # сохранение html в файл
    # lessons.html_soup.prettify()
    # lessons.save_html_file('test.html')
    # # сохранение файла с обработанными ссылками шагов
    # lessons.add_processed_steps()
    # lessons.write_processed_steps()
    #
    # if save_main_page:
    #     lessons.browser.close()

    task = MakeTask()
    task.auth_stepik()
    task.solution()
