# _*_ coding: utf-8 _*_

import psycopg2 as pg2
import os
import csv
from commom_module.articleHandler import Article, ArticleHtmlParser
from common_module.dirHandler import mkdir_p, del_folder
from common_module.dbHandler import dbHandler


USER_AGENT = 'Mozilla/5.0'
BASE_DIR = "./articles"
ORIGIN_PATH = os.path.join(BASE_DIR, 'Origin-Data')
NAVER_NEWS_URL_REGEX = "https?://news.naver.com"


def get_media_list_fromDB(db):

    sql = "SELECT DISTINCT media FROM news_list;"
    media_list = db.query(sql)

    return media_list


def get_article_list_fromDB(db, media_name):

    article_list = []
    sql = "SELECT DISTINCT media FROM news_list WHERE media = '{}';".format(media_name)
    rows = db.query(sql)

    for row in rows:
        article_list.append(Article(title=row[0], url=row[1], media=row[2]))

    return article_list


if __name__ == "__main__":
    del_folder(ORIGIN_PATH)
    mkdir_p(ORIGIN_PATH)
    
    db = dbHandler(host='localhost', dbname='newsdb', user='subinkim', password='123', port='5432')

    media_list = get_media_list_fromDB()
    for media_name in media_list :

        f = open(os.path.join(ORIGIN_PATH, media_name + ".csv"), 'w', newline="\n", encoding="utf-8")
        wr = csv.writer(f)

        article_list = get_article_list_fromDB(media_name)
        for article_no, article in enumerate(article_list):

            articleParser = ArticleHtmlParser(article.url, USER_AGENT)
            if not articleParser.checkUrlValid(NAVER_NEWS_URL_REGEX): continue
            print(articleParser.getUrl())

            try:
                content = articleParser.getArticleContent('div', 'articleBodyContents')
                article.setContent(content)
                wr.writerow([article.title, "\t".join(list(article.readContent()))])

            except Exception as e:
                print(e)
                pass
        f.close()

    media_list = os.listdir(ORIGIN_PATH)
    print("Media count {count}".format(count=len(media_list)))

    from functools import reduce
    print("Total Article Count {c}".format(c=reduce(lambda a,b : a+b,
                                                    [len(os.listdir(os.path.join(ORIGIN_PATH, media))) for media in media_list])))



