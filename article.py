import codecs
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict, Counter


class Article(object):
    stopwords = stopwords.words('english')

    def __init__(self, path, category):
        """

        :param path: Article Path
        :param category: Article Category
        """
        self.content = codecs.open(
            path, "r", encoding='utf-8', errors='ignore').readlines()
        self.title = self.content[0]
        self.body = ' '.join(self.content[1:])
        self.category = category
        self.title_tokens = self.lemmatize(self.tokenize(self.title))
        self.body_tokens = self.lemmatize(self.tokenize(self.body))
        self.tf = self.tf_index(self.body_tokens)

        self.tfidf = None
        self.tfidfie = None
        self.vector = None
        self.term_count()

    def document_terms(self):

        return self.terms

    def term_count(self):

        self.terms = Counter()

        for token in self.title_tokens:
            self.terms[token] += 1
        for token in self.body_tokens:
            self.terms[token] += 1

    def tokenize(self, text):
        """

        :param text:
        :return: text tokens
        """
        return [t.lower() for t in re.findall(r"\w+(?:[-']\w+)*", text) if t
                not in self.stopwords and len(t) > 2]

    def lemmatize(self, tokens):
        lemmatizer = WordNetLemmatizer()

        return [lemmatizer.lemmatize(token) for token in tokens]

    def tf_index(self, token_list):
        tf = defaultdict(lambda: 0)

        for token in token_list:
            tf[token] = tf[token] + 1

        return tf
