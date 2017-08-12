import math
from collections import defaultdict
import operator


class Classifier(object):
    def __init__(self):
        # stores (class, #articles)
        self.class_article_count = defaultdict(lambda: 0)
        # stores (class, prior)
        self.class_priors = defaultdict(lambda: 0)
        # stores vocab count of entire dataset
        self.vocab_count = None
        # stores (class,(term, freq)
        self.class_term_freq = defaultdict(lambda: defaultdict(lambda: 0))
        # stores (class, token_count)
        self.class_token_count = defaultdict(lambda: 0)

    def train(self, articles):
        """
        Given a list of articles, compute the class priors and class feature stats.
        """

        vocab_list = list()

        for article in articles:

            category = article.category
            tf = article.tf

            # count articles per category
            self.class_article_count[category] += 1

            for term in tf.keys():
                self.class_term_freq[category][term] = self.class_term_freq[category][term] + 1
                self.class_token_count[category] = self.class_token_count[category] + 1
                vocab_list.append(term)

        self.vocab_count = len(set(vocab_list))

        for key in self.class_article_count.keys():
            self.class_priors[key] = float(self.class_article_count[key]) / float(len(articles))

    def classify(self, articles):
        """
        Classify the list of articles.
        :param articles: The list of articles.
        :return: a list of strings, the class categories, for each article.
        """

        predictions = list()
        scores = defaultdict(lambda: 0)

        for article in articles:
            tf = article.tf

            for category, prior in self.class_priors.items():
                scores[category] = math.log10(prior)

                for token in tf.keys():
                    token_score = tf[token] * math.log10((self.class_term_freq[category][token] + 1) * 1.0 /
                                                         (self.class_token_count[category] + self.vocab_count))
                    scores[category] += token_score

            predictions.append(max(scores.items(), key=operator.itemgetter(1))[0])

            scores = dict.fromkeys(scores, 0)

        return predictions
