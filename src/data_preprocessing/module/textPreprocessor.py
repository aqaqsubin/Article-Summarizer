import re
from konlpy.tag import Komoran
from nltk.tokenize import RegexpTokenizer


class TextPreprocessor:
    def __init__(self):
        self.retokenize = RegexpTokenizer("[\w]+")
        self.swords = []
        self.tokenizer = {}
        self.tagger = Komoran()

    def removeDuplicateSpace(self, text):
        return re.sub('\s+', ' ', text)  # 중복 공백, 탭, 개행 제거
    
    def removeSpecialChar(self, text):
        return ' '.join(self.retokenize.tokenize(text))


    def loadSwords(self, filename):
        self.swords = []
        with open(filename, 'r', encoding='utf-8') as f:
            self.swords = f.readlines()
            self.swords = [sword.replace('\n','') for sword in self.swords]

        self.tokenizer = lambda sent: filter(lambda x:x not in self.swords, sent.split())

        return self.swords
        
    def removeSwords(self, text):
        return ' '.join([token for token in list(self.tokenizer(text))])
    
    def cleanLines(self, lines):
        return_val = list(map(lambda line: self.cleanLine(line), lines))
        return [line for line in return_val if line is not '']

    def cleanLine(self, line):
        cleanLine = self.removeDuplicateSpace(line)
        rmSpecial = self.removeSpecialChar(cleanLine)
        rmSwordLine = self.removeSwords(rmSpecial)
                    
        return rmSwordLine

    def del_personal_info(self, lines, media):
        result = []
        for line in lines:
            rmBracket = re.sub('(\([^)]*\)|\[[^]]*\])', '', line)  # 괄호 안 내용 제거
            rmMedia = re.sub(media, ' ', rmBracket)  # 언론사명 제거
            rmReporter = re.sub('[가-힣]{2,5}\s?기자', ' ', rmMedia) # 기자 이름 제거
            rmReporter = re.sub('[가-힣]{2,5}\s?특파원', ' ', rmReporter) # 기자 이름 제거
            rmEmail = re.sub('[0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_\.]?[0-9a-zA-Z])*\.[a-zA-Z]{2,3}', ' ', rmReporter) # 이메일 제거
            result.append(rmEmail)

        return result
