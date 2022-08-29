""" Загрузка количества вакансий и резюме по шаблонам"""

from parsing_hh import ParsingHH

hh_obj = ParsingHH()

for text_search in ('specialization', 'text', 'professional_role'):
    for kind_search in ('vacancy', 'resume'):
        print(f'Параметры поиска: kind={kind_search}, search={text_search}')
        hh_obj.getting_numbers_vacancy_resume(kind=kind_search,
                                              search=text_search)
