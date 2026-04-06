from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(corpus):
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(corpus).toarray()
    return X, tfidf