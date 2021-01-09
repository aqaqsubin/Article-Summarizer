# _*_ coding: utf-8 _*_

import psycopg2 as pg2
import requests
import os
from bs4 import BeautifulSoup, Comment
from shutil import rmtree

USER_AGENT = 'Mozilla/5.0'
CONTENT_DIR = "./articles"
NAVER_NEWS_BASE_URL = "news.naver.com"

def mkdir_p(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def del_folder(path):
    try:
        rmtree(path)
    except:
        pass

def get_article_list_fromDB():
    conn = None
    article_list = []
    try:
        conn = pg2.connect(host='localhost', dbname='newsdb', user='subinkim', password='123', port='5432')  # db에 접속
        cur = conn.cursor()

        cur.execute("SELECT title, url, media FROM news_list;")
        rows = cur.fetchall()
        for row in rows:
            article_list.append(dict(title=row[0], url=row[1], media=row[2]))

    except (Exception, pg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return article_list


def get_text_without_children(tag):
    return ''.join(tag.find_all(text=True, recursive=False)).strip()


def parse_article_content(article_html, tag, id):
    content = ''

    div = article_html.find(tag, id=id)
    for element in div(text=lambda text: isinstance(text, Comment)):
        element.extract()

    divs = article_html.find_all(tag, {"id": id})

    for i in divs:
        content += get_text_without_children(i)

    return content


def save_article_content_txt(article_no, title, content, media):
    f = open(os.path.join(CONTENT_DIR, str(article_no) + ".txt"), 'w', -1, "utf-8")
    f.write(title + '\n')
    f.write(content + '\n')
    f.write(media + '\n')
    f.close()


def get_article_content():
    article_list = get_article_list_fromDB()

    for article_no, article in enumerate(article_list):
        article_page_response = requests.get(article['url'], headers={'User-Agent': USER_AGENT})
        article_html = BeautifulSoup(article_page_response.text, "html.parser")

        url = article_html.find('meta', property='og:url')
        print(url['content'])
        if NAVER_NEWS_BASE_URL in url['content'] :
            content = parse_article_content(article_html, 'div', 'articleBodyContents')
            save_article_content_txt(article_no, article['title'], content, article['media'])

if __name__ == "__main__":
    mkdir_p(CONTENT_DIR)
    get_article_content()

