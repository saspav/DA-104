import re
import scrapy


class TutortopSpider(scrapy.Spider):
    """
    Паук для парсинга сайта tutortop.ru: сбор всех курсов с сайта
    """
    name = 'Tutortop'
    start_urls = ['https://tutortop.ru']

    def parse(self, response):
        for link in response.css('div.accordion-container .ac-a a.category_link[href*=courses_selection]::attr(href)'):
            yield response.follow(link.get(), callback=self.parse_page)

    @staticmethod
    def parse_page(response):
        fields = ['data-course', 'data-date', 'data-dlitelnost',
                  'data-pd_price',
                  'data-popularity', 'data-price', 'data-rassrochka',
                  'data-rating',
                  'data-school', 'data-school_reviews']

        category = response.css('.breadcrumbs a[href*=courses_category]::text')[0].get()
        partition = response.css('.breadcrumb_last::text')[0].get()
        courses = response.css('.tab-course-items-sticky .tab-course-item')

        for course in courses:
            course_title = course.css('.m-course-title::text').get()
            course_row = dict(category=category, partition=partition,
                              course_title=course_title)
            str_course = str(course.get())
            for field in fields:
                pattern = r'(?<=' + field + r'=\").*?(?=\")'
                find = re.findall(pattern, str_course)
                found = find[0] if find else ''
                course_row[field] = found

            url = course.css('a.tab-link-course[href]::attr(href)').get()
            course_row['url'] = url

            yield course_row
