# _*_ coding: utf-8 _*_

import psycopg2 as pg2
import requests
import re
import os
from bs4 import BeautifulSoup, Comment
from shutil import rmtree

USER_AGENT = 'Mozilla/5.0'
BASE_DIR = "../articles"
ORIGIN_PATH = os.path.join(BASE_DIR, 'Origin-Data')
NAVER_NEWS_URL_REGEX = "https?://news.naver.com"


class Article:
    def __init__(self, title, media, url):
        self.title = title
        self.media = media
        self.url = url
        self.content = ''

    def setContent(self, content):
        self.content = content

    def saveArticle(self, baseDir):
        path = os.path.join(baseDir, self.media)
        mkdir_p(path)

        files = os.listdir(path)

        f = open(os.path.join(path, str(len(files)) + ".txt"), 'w', -1, "utf-8")
        f.write(self.title + '\n')
        f.write(self.content + '\n')
        f.write(self.media + '\n')
        f.close()

class ArticleHtmlParser:

    def __init__(self, url, userAgent):
        article_page_response = requests.get(url, headers={'User-Agent': userAgent})
        self.article_html = BeautifulSoup(article_page_response.text, "html.parser")
        self.redirectedUrl = self.article_html.find('meta', property='og:url')

    def checkUrlValid(self, regEx):
        try:
            return re.match(regEx, self.redirectedUrl['content']) is not None
        except Exception:
            return False

    def getUrl(self):
        return self.redirectedUrl['content']

    def get_text_without_children(self, tag):
        return ''.join(tag.find_all(text=True, recursive=False)).strip()

    def getArticleContent(self, tag, id):
        content = ''

        div = self.article_html.find(tag, id=id)
        if div is None:
            raise Exception("Page Not Found")

        for element in div(text=lambda text: isinstance(text, Comment)):
            element.extract()

        divs = self.article_html.find_all(tag, {"id": id})

        for i in divs:
            content += self.get_text_without_children(i)

        return content



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
            article_list.append(Article(title=row[0], url=row[1], media=row[2]))

    except (Exception, pg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return article_list


if __name__ == "__main__":
    del_folder(ORIGIN_PATH)
    mkdir_p(ORIGIN_PATH)

    article_list = get_article_list_fromDB()

    for article_no, article in enumerate(article_list):

        articleParser = ArticleHtmlParser(article.url, USER_AGENT)

        if not articleParser.checkUrlValid(NAVER_NEWS_URL_REGEX): continue
        print(articleParser.getUrl())

        try:
            content = articleParser.getArticleContent('div', 'articleBodyContents')
            article.setContent(content)
            article.saveArticle(ORIGIN_PATH)

        except Exception as e:
            print(e)
            pass

    media_list = os.listdir(ORIGIN_PATH)
    print("Media count {count}".format(count=len(media_list)))

    from functools import reduce
    print("Total Article Count {c}".format(c=reduce(lambda a,b : a+b,
                                                    [len(os.listdir(os.path.join(ORIGIN_PATH, media))) for media in media_list])))



