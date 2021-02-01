import re
import requests

from bs4 import BeautifulSoup, Comment

class Article:
    def __init__(self, title, media, url):
        self.title = title
        self.media = media
        self.url = url
        self.content = ''
        self.rgxSplitter = re.compile('([.!?:-](?:["\']|(?![0-9])))')

    def setContent(self, content):
        self.content = content

    def readContent(self):
        self.content = del_personal_info()
        docs = self.rgxSplitter.split(self.content)
            
            if not is_splited_sentence(docs): # 본문이 1줄이며, 위 정규식에 따라 split 되지 않음
                yield docs[0]
            else :
                for s in map(lambda a, b: a + b, docs[::2], docs[1::2]):
                    if not s: continue
                    yield s

    def del_personal_info(self):
        rmBracket = re.sub('(\([^)]*\)|\[[^]]*\])', '', self.content)  # 괄호 안 내용 제거
        rmMedia = re.sub(self.media, ' ', rmBracket)  # 언론사명 제거
        rmReporter = re.sub('[가-힣]{2,5}\s?기자', ' ', rmMedia) # 기자 이름 제거
        rmEmail = re.sub('[0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*\.[a-zA-Z]{2,3}', ' ', rmReporter) # 이메일 제거

        self.content = rmEmail


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
