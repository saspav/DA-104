import scrapy


class ZakvaskiSpider(scrapy.Spider):
    name = 'Zakvaski'
    start_urls = ['https://pro-syr.ru/zakvaski-dlya-syra/mezofilnye/']

    def parse(self, response):
        for link in response.css('div.product-layout a::attr(href)'):
            yield response.follow(link.get(), callback=self.parse_page)

        last_page = response.css('ul.pagination a::attr(href)')[-1].get()
        last_page = int(last_page.split('=')[-1])
        for num_page in range(2, last_page + 1):
            link = f'https://pro-syr.ru/zakvaski-dlya-syra/mezofilnye/?page={num_page}'
            yield response.follow(link, callback=self.parse)

    @staticmethod
    def parse_page(response):
        price = response.css('.autocalc-product-price::text').get()
        price = price.split('руб')[0].replace(' ', '')
        yield {
            "name": response.css('#content h1::text').get(),
            "price": price,
            "in_stock": response.css('b.outstock::text').get()
        }
