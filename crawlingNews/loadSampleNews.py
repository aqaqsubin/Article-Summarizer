import os
from loadNews import mkdir_p, del_folder
import random

BASE_DIR = "./articles"
SAMPLE_BASE_DIR = "./sample_articles"
ORIGIN_PATH = os.path.join(BASE_DIR, 'Origin-Data')

class ArticleCopier:
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()

    def copy_article(self, save_dir):
        mkdir_p(save_dir)
        article_name = str(len(os.listdir(save_dir)))+".txt"

        with open(os.path.join(save_dir, article_name), 'w', encoding='utf-8') as new_f:
            new_f.writelines(self.lines)

def get_entire_news_list():
    entire_list = []
    media_list = os.listdir(ORIGIN_PATH)

    for media in media_list:
        media_path = os.path.join(ORIGIN_PATH, media)
        article_list = os.listdir(media_path)

        entire_list += [os.path.join(media_path, article_name) for article_name in article_list]

    return entire_list

def get_sample_news(news_num, save_root_dir):

    entire_list = get_entire_news_list()

    # 랜덤으로 max_count만큼 기사 샘플 수집
    random.shuffle(entire_list)

    # 수집 가능한 샘플 개수
    max_count = news_num if news_num <= len(entire_list) else len(entire_list)
    print("수집 가능한 샘플 개수  : {max_count}".format(max_count=max_count))
    sample_list = [entire_list[idx] for idx in range(max_count)]

    for sample in sample_list:
        path_token = sample.split('/')
        path_token[1] = save_root_dir.split('/')[1]
        save_dir_path = '/'.join(path_token[:4])

        article = ArticleCopier(sample)
        article.copy_article(save_dir_path)

    print("샘플 수집 완료")

if __name__ == "__main__" :
    save_root_dir = os.path.join(SAMPLE_BASE_DIR, 'Origin-Data')

    del_folder(save_root_dir)
    mkdir_p(save_root_dir)

    # 100,000개 샘플 기사 수집
    get_sample_news(100000, save_root_dir)

    media_list = os.listdir(save_root_dir)
    print("Media count {count}".format(count=len(media_list)))

    from functools import reduce
    print("Total Article Count {c}".format(c=reduce(lambda a, b: a + b,
                                                    [len(os.listdir(os.path.join(save_root_dir, media))) for media in
                                                     media_list])))