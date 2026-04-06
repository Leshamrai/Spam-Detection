import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_text
from src.feature_extraction import extract_features
from src.train_model import train_models
from src.evaluate import evaluate_model
from src.predict import predict

df = pd.read_csv('dataset/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df['processed'] = df['message'].apply(preprocess_text)

X, tfidf = extract_features(df['processed'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

nb_model, lr_model = train_models(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

print("Naive Bayes:", evaluate_model(y_test, y_pred_nb))
print("Logistic Regression:", evaluate_model(y_test, y_pred_lr))

msg = "You won a free prize!"
print("Prediction:", predict(msg, nb_model, tfidf, preprocess_text))

def train_and_get_models():
    return nb_model, lr_model, tfidf, y_test, y_pred_nb, y_pred_lr