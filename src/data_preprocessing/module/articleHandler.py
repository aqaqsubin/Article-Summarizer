import re

class Article:
    def __init__(self, title, media, contents):
        self.title = title
        self.media = media
        self.contents = contents

    def readContent(self):
        for line in self.contents[:-1]:
            if line is '': continue
            yield line
        