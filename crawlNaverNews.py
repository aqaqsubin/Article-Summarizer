# _*_ coding: utf-8 _*_

import requests
import os
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from loadNews import get_text_without_children

NAVER_NEWS_BASE_URL = "news.naver.com"
NAVER_SEARCH_URL = "https://search.naver.com/search.naver"

NAVER_SEARCH_POLICE_URL = "https://search.naver.com/search.naver?&where=news&pd=3&query=경찰&ds={ds}&de={de}"

USER_AGENT = 'Mozilla/5.0'

SQL_FILE_PATH = 'insertSqlQuery.sql'
SQL_INSERT_BASE_QUERY = "INSERT INTO {table_name} (title, media, url) VALUES (%s, %s, %s);"


def get_append_option(filename):
    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    return append_write


def get_search_page_list(ds, de):

    search_page_list = []

    url = NAVER_SEARCH_POLICE_URL.format(ds=ds.strftime("%Y.%m.%d"), de=de.strftime("%Y.%m.%d"))
    while True:
    # for i in range(10*1000):
        try:
            search_response = requests.get(url, headers={'User-Agent': USER_AGENT})
            search_html = BeautifulSoup(search_response.text, "html.parser")

            next_page_url = search_html.select_one("#main_pack > div.api_sc_page_wrap > div > a.btn_next")["href"]
            print(next_page_url)
            search_page_list.append(NAVER_SEARCH_URL + next_page_url)

            url = search_page_list[-1]

        except KeyError as keyErr:
            break

    return search_page_list;


def parse_article_info(article):
    title = article.select_one("a.news_tit")["title"]

    divs = article.find_all('a', {"class": "info press"})
    try:
        media = get_text_without_children(divs[0])
    except Exception as e :
        raise("Error Appeared : ", e)
    src_addr = article.select_one("div.news_info > div.info_group > a:last-of-type")["href"]

    return title, media, src_addr


def insert_to_db():
    try:
        os.system("set PGPASSWORD=123")
        os.system("psql -U subinkim newsdb < " + SQL_FILE_PATH)
        os.remove(SQL_FILE_PATH)
    except Exception:
        pass

def convert_txt(text): # Convert Single Quote -> Double Quote, Wrapping Text with Single Quote
    return "\'"+text.replace("\'","\"")+"\'"

def save_article_info(article_info):

    if not NAVER_NEWS_BASE_URL in article_info['src_addr']: return

    baseQuery = "INSERT INTO news_list (title, media, url) VALUES ({title}, {media}, {url});"
    query = baseQuery.format(title=article_info['title'], media=article_info['media'],
                             url=article_info['src_addr'])

    f = open(SQL_FILE_PATH, get_append_option(SQL_FILE_PATH))
    f.write(query + "\n")
    f.close()

    fileStat = os.stat(SQL_FILE_PATH)
    fileSize = fileStat.st_size

    if fileSize > 100 * 1000: insert_to_db()

def get_article_list():

    ds = datetime.now()
    de = datetime.now()

    while True:
        search_page_list = get_search_page_list(ds, de)
        for page_no, page_url in enumerate(search_page_list):

            page_response = requests.get(page_url, headers={'User-Agent': USER_AGENT})
            page_html = BeautifulSoup(page_response.text, "html.parser")


            article_area_list = page_html.select("ul.list_news > li.bx ")
            for idx, article_area in enumerate(article_area_list):
                article = article_area_list[idx].select_one("div.news_wrap.api_ani_send > div.news_area")
                title, media, src_addr = parse_article_info(article)

                article_info = {'title': convert_txt(title),
                                'media': convert_txt(media),
                                'src_addr': convert_txt(src_addr)}

                save_article_info(article_info)

        ds += timedelta(days=-1)
        de += timedelta(days=-1)

    insert_to_db()

if __name__ == "__main__":
    get_article_list()
