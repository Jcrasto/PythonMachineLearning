import urllib.request
import pandas
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_colwidth', None)


n_comp = 50
no_top_words = 15
no_features = 1000

def list_topics(model,feature_names,no_top_words):
    topiclist =[]
    for topic_idx, topic in enumerate(model.components_):
        topiclist.append(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    return(topiclist)


def main(booklist):
    linelist = []
    for book in booklist:
        text = urllib.request.urlopen(book)
        for line in text:
            txt = line.rstrip().decode("utf-8")
            if len(txt.split()) > 3:
                linelist.append(txt)

    cv = CountVectorizer(stop_words='english')
    tf = cv.fit_transform(linelist)
    tf_feature_names = cv.get_feature_names()

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(linelist)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_components=n_comp, max_iter=5, learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tf)
    Topic_Model = pandas.DataFrame(list_topics(lda, tf_feature_names, no_top_words), columns= ["cv_lda"])

    nmf = NMF(n_components=n_comp, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tf)
    Topic_Model['cv_nmf'] = pandas.DataFrame(list_topics(nmf, tf_feature_names, no_top_words))

    lda = LatentDirichletAllocation(n_components=n_comp, max_iter=5, learning_method='online', learning_offset=50.,
                                    random_state=0).fit(tfidf)
    Topic_Model['tfidf_lda'] = pandas.DataFrame(list_topics(nmf, tf_feature_names, no_top_words))

    nmf = NMF(n_components=n_comp, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    Topic_Model['tfidf_nmf'] = pandas.DataFrame(list_topics(nmf, tfidf_feature_names, no_top_words))

    print(Topic_Model.head())


if __name__ == '__main__':
    booklist = ["https://www.gutenberg.org/files/215/215-0.txt","https://www.gutenberg.org/files/1342/1342-0.txt",
                "https://www.gutenberg.org/files/84/84-0.txt","http://www.gutenberg.org/cache/epub/2542/pg2542.txt",
                "http://www.gutenberg.org/cache/epub/1064/pg1064.txt", "http://www.gutenberg.org/cache/epub/376/pg376.txt"]
    # Call of the Wild, Pride and Prejudice, Frankenstein, A Dolls House, The Masque of the Red Death, A Journal of the Plague Year
    booklist = ["https://www.gutenberg.org/files/215/215-0.txt"]
    main(booklist)

