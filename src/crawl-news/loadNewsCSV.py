# _*_ coding: utf-8 _*_

import psycopg2 as pg2
import os
import csv
import pandas as pd
from common_module.articleHandler import Article, ArticleHtmlParser
from common_module.dirHandler import mkdir_p, del_folder
from common_module.dbHandler import dbHandler


USER_AGENT = 'Mozilla/5.0'
BASE_DIR = "./articles"
ORIGIN_PATH = os.path.join(BASE_DIR, 'Origin-Data')
NAVER_NEWS_URL_REGEX = "https?://news.naver.com"


def get_media_list_fromDB(db):

    sql = "SELECT DISTINCT media FROM news_list;"
    media_list = db.query(sql)
    media_list = [media_tuple[0] for media_tuple in media_list]

    return media_list


def get_article_list_fromDB(db, media_name):

    article_list = []
    sql = "SELECT * FROM news_list WHERE media = '{}';".format(media_name)
    rows = db.query(sql)

    for row in rows:
        article_list.append(Article(title=row[0], url=row[1], media=row[2]))

    return article_list

def saveCSVFile(baseDir, media, article_dist):
    save_path = os.path.join(baseDir, media) + ".csv"
    article_dist.to_csv(save_path, mode='w', header=False)

if __name__ == "__main__":
    del_folder(ORIGIN_PATH)
    mkdir_p(ORIGIN_PATH)
    
    db = dbHandler(host='localhost', dbname='newsdb', user='subinkim', password='123', port='5432')

    media_list = get_media_list_fromDB(db)

    for media_name in media_list :
        print(media_name)

        article_list = get_article_list_fromDB(db, media_name)
        article_dist = pd.DataFrame(columns=['Title', 'Contents'])
        for article_no, article in enumerate(article_list):

            articleParser = ArticleHtmlParser(article.url, USER_AGENT)
            if not articleParser.checkUrlValid(NAVER_NEWS_URL_REGEX): continue
            print(articleParser.getUrl())

            try:
                content = articleParser.getArticleContent('div', 'articleBodyContents')
                article.setContent(content)
                conts_list = list(article.readContent())

                if not conts_list : continue
                dist = {'Title': article.title, 'Contents': "\t".join(conts_list)}
                article_dist = article_dist.append(dist, ignore_index=True)

            except Exception as e:
                print(e)
                pass
        saveCSVFile(ORIGIN_PATH, media_name, article_dist)

    media_list = os.listdir(ORIGIN_PATH)
    print("Media count {count}".format(count=len(media_list)))



