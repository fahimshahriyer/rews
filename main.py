import random
from collections import defaultdict, Counter
import os
from article import Article
from index import Index
from classifier import Classifier
from recommender import KNN


def recommend(articles, test_articles):
    print("\n\n\nNews Recommendation System")
    print("--------------------------")

    # ask user for the desired option count and recommendation count. set default value in case invalid inputs.
    try:
        option_count = int(
            input("\nEnter number of articles to choose from. [number from 5 to 10 suggested]: "))
        if option_count < 1 or option_count > 20:
            print("Invalid Choice.. By default selected 5.")
            option_count = 5
    except:
        print("Invalid Choice.. By default selected 5.")
        option_count = 5

    try:
        k_n = int(
            input("\nEnter number of recommendation per article. [number from 5 to 10 suggested]: "))
        if k_n < 1 or k_n > 20:
            print("Invalid Choice.. By default selected 5.")
            k_n = 5
    except:
        print("Invalid Choice.. By default selected 5.")
        k_n = 5

    end = False

    # run the loop until user quits.
    while not end:

        # pick random documents from test docs and provide titles to the user.
        choice_list = random.sample(test_articles, option_count)

        while True:
            print("\n---Available Choices For Articles(Titles)---\n")

            for i in range(len(choice_list)):
                print(str(i + 1) + ": " + choice_list[i].title)

            print("r: Refresh List")
            print("q: Quit()\n")

            choice = input("Enter Choice: ")

            if choice == 'q':
                end = True
                break
            elif choice == 'r':
                break
            else:
                try:
                    user_choice = int(choice) - 1
                    if user_choice < 0 or user_choice >= len(choice_list):
                        print("Invalid Choice.. Try Again..")
                        continue
                except:
                    print("Invalid Choice.. Try Again..")
                    continue
                selected_article = choice_list[user_choice]

                prediction_list = list()

                classifier = Classifier()

                prediction_list.append(classifier.classify([selected_article])[0])

                prediction_count = Counter(prediction_list)
                top_prediction = prediction_count.most_common(1)

                if top_prediction[0][1] > 1:
                    prediction = top_prediction[0][0]
                else:
                    prediction = prediction_list[0]

                # create knn instance using documents of predicted topic. and find k closest documents.
                knn = KNN(articles[prediction])
                k_neighbours = knn.find_k_neighbours(selected_article, k_n)

                while True:
                    print("\nRecommended Articles for : " + selected_article.title)
                    for i in range(len(k_neighbours)):
                        print(str(i + 1) + ": " + k_neighbours[i].title)
                    next_choice = input("\nEnter Next Choice: [Article num to read the article. "
                                        "'o' to read the original article. "
                                        "'b' to go back to article choice list.]  ")

                    if next_choice == 'b':
                        break
                    elif next_choice == 'o':
                        text = selected_article.text
                        print("\nArticle Text for original title : " + selected_article.title)
                        print(text)
                    else:
                        try:
                            n_choice = int(next_choice) - 1
                            if n_choice < 0 or n_choice >= k_n:
                                print("Invalid Choice.. Try Again..")
                                continue
                        except:
                            print("Invalid Choice.. Try Again..")
                            continue
                        text = k_neighbours[n_choice].text
                        print("\nArticle Text for recommended title : " +
                              k_neighbours[n_choice].title)
                        print(text)


def main():
    category_path = "dataset/"

    articles = defaultdict(lambda: list())

    categories = list()

    print("Reading all articles in the Dataset")

    for category in filter(lambda f: not f.startswith('.'), os.listdir(category_path)):
        articles_path = category_path + category + '/'
        categories.append(category)
        temp_articles = list()

        for article in os.listdir(articles_path):
            article_path = articles_path + article
            temp_articles.append(Article(article_path, category))

        articles[category] = temp_articles[:]

    training_articles, test_articles = list(), list()

    fold_count = 10

    for key, value in articles.items():
        random.shuffle(value)
        test_len = int(len(value) / fold_count)
        training_articles += value[:-test_len]
        test_articles += value[-test_len:]

    index = Index(training_articles)

    print("Total training article : " + str(len(training_articles)))
    print("Total test article : " + str(len(test_articles)))

    test_categories = [a.category for a in test_articles]

    for article in training_articles:
        article.vector = article.tfidfie

    for article in test_articles:
        article.vector = article.tf

    classifier = Classifier()
    print("Training...\n")
    classifier.train(training_articles)

    predictions = classifier.classify(test_articles)

    recommend(articles, test_articles)


if __name__ == '__main__':
    main()
