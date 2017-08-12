from collections import defaultdict
import math
import operator


class Index(object):
    def __init__(self, articles):

        self.articles = articles
        self.tf = [articles.tf for articles in self.articles]

        self.tf_index = self.create_tf_index(self.tf)
        self.article_freqs = self.count_article_frequencies(self.tf_index)
        self.tfidf_list = self.create_tfidf_index(self.tf, self.article_freqs)
        self.normal_tfidf_list = self.normalize_vector_list(self.tfidf_list)
        self.update_tfidf(self.articles, self.normal_tfidf_list)

        self.category_article_freqs = self.count_category_article_frequencies(self.articles)
        self.information_entropy = self.cal_information_entropy(self.article_freqs, self.category_article_freqs)
        self.tfidfie_list = self.create_tfidfie_index(self.tfidf_list, self.information_entropy)
        self.normal_tfidfie_list = self.normalize_vector_list(self.tfidfie_list)
        self.update_tfidfie(self.articles, self.normal_tfidfie_list)

    def create_tf_index(self, tf):
        tf_index = defaultdict(lambda: list())

        for i in range(len(tf)):
            article = tf[i]

            for term, freq in article.items():
                tf_index[term].append([i, freq])
        return tf_index

    def count_article_frequencies(self, tf_index):
        article_freqs = defaultdict(lambda: 0)

        for term, freq_list in tf_index.items():
            article_freqs[term] = len(freq_list)

        return article_freqs

    def create_tfidf_index(self, tf, article_freqs):
        article_count = len(tf)
        tfidf_list = list()

        for article_tf in tf:
            article_tfidf = defaultdict(lambda: 0.0)

            for term, freq in article_tf.items():
                score = freq * math.log(article_count * 1.0 / article_freqs[term])
                article_tfidf[term] = score

            tfidf_list.append(article_tfidf)

        return tfidf_list

    def normalize_vector_list(self, vector_list):
        normal_vector_list = vector_list

        for vector in vector_list:
            vector_normal = self.cosine_normalization(vector.values())

            for key in vector.keys():
                vector[key] = vector[key] / vector_normal

        return normal_vector_list

    def cosine_normalization(self, vector):
        return math.sqrt(sum(i ** 2 for i in vector))

    def update_tfidf(self, articles, tfidf_list):
        for i in range(len(articles)):
            articles[i].tfidf = dict(sorted(tfidf_list[i].items(), key=operator.itemgetter(1), reverse=True)[:30])

    def count_category_article_frequencies(self, articles):
        category_article_freqs = defaultdict(lambda: defaultdict(lambda: 0))

        for article in articles:
            category = article.category

            for term in article.tf.keys():
                category_article_freqs[category][term] += 1

        return category_article_freqs

    def cal_information_entropy(self, article_freqs, category_article_freqs):
        information_entropy = defaultdict(lambda: 0.0)

        for term in article_freqs.keys():
            score = 0.0

            for category in category_article_freqs.keys():

                if category_article_freqs[category][term] != 0:
                    temp = category_article_freqs[category][term] * 1.0 / article_freqs[term]
                    score -= (temp * math.log(temp))

            if score == 0:
                score = 1.0

            information_entropy[term] = score

        return information_entropy

    def create_tfidfie_index(self, tfidf_list, information_entropy):
        tfidfie_list = list()
        for item in tfidf_list:
            tfidfie_list.append(item.copy())

        for article in tfidfie_list:
            for term in article.keys():
                article[term] = article[term] / information_entropy[term]

        return tfidfie_list

    def update_tfidfie(self, articles, tfidfie_list):
        for i in range(len(articles)):
            articles[i].tfidfie = dict(sorted(tfidfie_list[i].items(), key=operator.itemgetter(1), reverse=True)[:30])
