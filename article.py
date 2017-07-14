from nltk.corpus import stopwords
import codecs
import re


class Article(object):

    def __init__(self, path):
        self.content = codecs.open(path, "r", encoding='utf-8', errors='ignore').readlines()
        self.title = self.content[0]
        self.body = ' '.join(self.content[1:])