import os
import re
import pandas as pd
import scipy.stats as stats
from bs4 import BeautifulSoup
from datetime import datetime
from parsing_hh import ParsingHH
from stepik_addons import MakeHTML


class ParsingTT(ParsingHH):
    def __init__(self):
        """
        Инициализация экземпляра объекта
        """
        super().__init__()

        # ДФ со списком разделов курсов
        self.df_cs = pd.DataFrame()
        # ДФ со списком курсов
        self.df_courses = pd.DataFrame()
        # ДФ со предобработанными курсами
        self.df_courses_proc = pd.DataFrame()
        self.name_file_courses = os.path.join(self.path_file,
                                              'parsing_tutortop.csv')
        self.courses_selection = os.path.join(self.path_file,
                                              'courses_selection.csv')
        self.processed_courses = os.path.join(self.path_file,
                                              'processed_tutortop.csv')
        # Создание экземпляра класса из Степика
        self.step_obj = MakeHTML(base_dir=self.path_file)

    def get_courses_from_page(self, url=None, file_save=False, file_read=None):
        """
        Загрузка странцы с курсами по одному разделу
        :param url: ссылка на раздел курсов
        :param file_save: сохранять скачанную страницу в .html файл
        :param file_read: прочитать ранее скачанную страницу из файла:
        имя файла без пути или True - тогда имя файла будет сформировано из url
        :return: содержимое скачанной страницы - html text
        """
        name_html = ''
        html_text = ''

        if url is not None:
            name_html = f"{url.strip('/').split('/')[-1]}.html"

        if file_read is not None:
            if isinstance(file_read, str):
                name_html = file_read
            if name_html:
                html_text = self.step_obj.read_html_file(name_html)

        elif url is not None:
            html = self.download_page(url, ret_json=False)
            html_text = html.content
            # print(html_text)

            if file_save and name_html:
                # сохранение в файл для опытов
                self.step_obj.save_html_file(name_html, html_text=html_text)

        return html_text

    @staticmethod
    def re_encoding_str(text):
        if isinstance(text, str):
            return text.strip()
        return text

    def soup_partition_courses(self, html_text):
        """
        Получение информации о курсах по файлу html_text (обычно один раздел)
        :param html_text:
        :return: ДФ с данными
        """

        fields = ['data-course', 'data-date', 'data-dlitelnost',
                  'data-pd_price',
                  'data-popularity', 'data-price', 'data-rassrochka',
                  'data-rating',
                  'data-school', 'data-school_reviews']

        df_partition_columns = ['category', 'partition', 'course_title']
        df_partition_columns.extend(field.split('-')[-1] for field in fields)
        df_partition_columns.extend(['discount', 'course_url'])

        dfp = pd.DataFrame(columns=df_partition_columns)

        soup = BeautifulSoup(html_text, 'lxml')

        breadcrumbs = soup.select('.breadcrumbs a[href*=courses_category]')[0]
        category = breadcrumbs.text
        partition = soup.select('.breadcrumb_last')[0].text
        print(f'Категория: "{category}", Раздел: "{partition}"', '\n')

        courses = soup.select(
            '.tab-course-items.tab-course-items-sticky .tab-course-item')
        payed_count = len(courses)
        courses = soup.select('.tab-course-items-sticky .tab-course-item')
        print('Всего курсов:', len(courses))
        print('Платных курсов:', payed_count, '\n')

        for course in courses:
            # print(course)
            # print()
            course_title = course.select('.m-course-title')[0].text
            course_row = [category, partition, course_title]
            str_course = str(course)
            for field in fields:
                pattern = r'(?<=' + field + r'=\").*?(?=\")'
                find = re.findall(pattern, str_course)
                found = find[0] if find else ''
                # print(f'{field} = {found}')
                course_row.append(found)
            rebate = course.select('.price__container--with-discount')
            discount = 0
            if rebate:
                rebate = rebate[0]
                # for price in rebate.select('span'):
                #     print(price.text, ''.join(re.findall('\d+', price.text)))
                discount = [''.join(re.findall('\d+', s.text))
                            for s in rebate.select('span')][-1]
                # print('Со скидкой:', discount)

            # прошлая версия
            # url = str(course.select('.m-course-school-link')[0])
            # url = re.findall(r'(?<=href=\")/goto/\?number=\d*&?', url)[0]
            # url = f"https://tutortop.ru{url}term=9"

            url = ''
            links = course.select('a.tab-link-course[href]')
            if links:
                link = links[0]
                if link.has_attr('href'):
                    url = link['href'].strip()
            print(f"Курс: {course_title.strip()}. Ссылка: {url}")
            # print(course_row)

            course_row.extend([discount, url])
            course_row = [self.re_encoding_str(item) for item in course_row]
            dfp.loc[len(dfp)] = [*course_row]

        return dfp

    def make_list_courses(self):
        """
        Формирование списка курсов по разделам в ДФ и сохрание в файл .csv
        :return: None
        """
        name_html = 'TutorTop.ru - агрегатор-отзовик всех онлайн-курсов.html'
        html_text = self.step_obj.read_html_file(name_html)
        soup = BeautifulSoup(html_text, 'lxml')
        links = soup.select('a.category_link[href*=courses_selection]')
        # # отображение названий и ссылок на разделы курсов
        # for link in links:
        #     if link.has_attr('href'):
        #         print(link.text, link['href'])
        #
        links = [(link.text.strip(), link['href'].strip()) for link in links
                 if link.has_attr('href')]
        # создание ДФ из разделов
        self.df_cs = pd.DataFrame(links,
                                  columns=['partition_name', 'partition_url'])
        # сохранение ДФ в файл
        self.df_cs.drop_duplicates(inplace=True, ignore_index=True)
        self.df_cs.to_csv(self.courses_selection, sep=';', index=False)

    def read_list_courses(self):
        """
        Чтение из файла .csv в ДФ списка курсов по разделам
        :return: None
        """
        if os.access(self.courses_selection, os.F_OK):
            self.df_cs = pd.read_csv(self.courses_selection, sep=';')
        else:
            print(f'Отсутствует файл {self.courses_selection}')

    def read_df_courses(self):
        # Чтение из файла данных о загруженных курсах
        if os.access(self.name_file_courses, os.F_OK):
            self.df_courses = pd.read_csv(self.name_file_courses, sep=';')

    def load_all_courses(self):
        """
        Загрузка курсов со всех страниц и сохранение ДФ в файл .csv
        :return: None
        """
        # Загрузка лог-файла по обработанным url из списка разделов курсов
        file_log = self.name_file_courses.replace('.csv', '.log')
        if os.access(file_log, os.F_OK):
            log = pd.read_csv(file_log, header=None)
            log.columns = ['url']
        else:
            log = pd.DataFrame(columns=['url'])
        # создание множества обработанных url
        log_urls = set(log.url.values)
        # print(log_urls)

        # Чтение из файла данных о загруженных курсах
        self.read_df_courses()

        # Чтение из файла .csv в ДФ списка курсов по разделам
        self.read_list_courses()

        for row in self.df_cs.itertuples(index=False):
            url = row[1]
            if url in log_urls:
                # url уже отработан
                continue
            else:
                print(f'Загружаю: {url}')

            html_txt = self.get_courses_from_page(url=url)
            data = self.soup_partition_courses(html_txt)
            if len(self.df_courses):
                self.df_courses = pd.concat([self.df_courses, data],
                                            ignore_index=True)
            else:
                self.df_courses = data
            print(f'В датафрейме {len(self.df_courses.index)} записей')
            # сохранение в файл
            self.df_courses.to_csv(self.name_file_courses, sep=';',
                                   index=False)
            # запишем в лог обработанный url
            with open(file_log, 'a+') as fw:
                fw.write(f'{url}\n')

    def preprocess_df_courses(self):
        # Чтение из файла данных о загруженных курсах
        self.read_df_courses()
        df = self.df_courses

        # почистим данные
        df.drop(['popularity', 'pd_price'], axis=1, inplace=True)
        # заполним пропуски нулями
        df.fillna(0, inplace=True)
        # преобразуем в целое число
        for name_col in ('price', 'rassrochka'):
            df[name_col] = df[name_col].astype(int)
        # приведение отсутствии рассрочки к одному виду
        df.loc[df['rassrochka'] == 1, 'rassrochka'] = 0
        # преобразование даты из unixtime формата в читаемую дату
        df['date'] = df['date'].apply(
            lambda x: datetime.fromtimestamp(x).date())
        # заполнение пропусков в датах на 01.09.2021
        date_first = datetime(2021, 9, 1).date()
        df.loc[df['date'] < date_first, 'date'] = date_first
        # преобразование длительности курса в число
        df['dlitelnost'] = df['dlitelnost'].str.replace(',', '.')
        df['dlitelnost'] = pd.to_numeric(df['dlitelnost'])
        # формирование полного url
        df['course_url'] = df['course_url'].apply(
            lambda x: f'https://tutortop.ru{x}')
        # переименование колонок
        df.rename(columns={'course': 'course_id', 'date': 'date_begin',
                           'dlitelnost': 'duration'},
                  inplace=True)

        # отфильтруем категорию курсов, которые не интересны заказчику
        # и бесплатные курсы
        out_category = ('Детям', 'Образ жизни', 'Создание контента')
        flt_category = (~df.category.isin(out_category)) & (df.price > 0)
        df = df[flt_category]
        df = df[df.price < 420000]
        # Получение лидеров (школ) по количеству курсов
        dfp = df.drop(['category', 'partition'],
                      axis=1).drop_duplicates('course_id')
        s_counts = dfp['school'].value_counts()[:13][::-1]
        # print(s_counts)
        # отметка ТОП-13 школ
        df['top_school'] = df.school.isin(s_counts.index)

        tmp = df[['course_id', 'course_url']]
        tmp.drop_duplicates(inplace=True)
        tmp.to_csv(self.processed_courses.replace('.csv', '_courses.csv'),
                   sep=';', index=False)

        df.to_csv(self.processed_courses, sep=';', index=False)
        df.to_excel(self.processed_courses.replace('.csv', '.xlsx'),
                    index=False)

    def read_df_processed_courses(self):
        # Чтение из файла данных об обработанных курсах
        if os.access(self.processed_courses, os.F_OK):
            self.df_courses_proc = pd.read_csv(self.processed_courses, sep=';')


if __name__ == "__main__":
    tt_obj = ParsingTT()

    # Чтение из файла данных о загруженных курсах
    tt_obj.preprocess_df_courses()
    tt_obj.read_df_processed_courses()
    df = tt_obj.df_courses_proc
    print(df.columns)
    print(df.info())
