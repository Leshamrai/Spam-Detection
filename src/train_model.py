from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def train_models(X_train, y_train):
    nb_model = MultinomialNB()
    lr_model = LogisticRegression(max_iter=1000)

    nb_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    return nb_model, lr_model