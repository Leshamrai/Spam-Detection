def predict(text, model, vectorizer, preprocess_func):
    text = preprocess_func(text)
    vector = vectorizer.transform([text]).toarray()
    prediction = model.predict(vector)

    return "Spam" if prediction[0] == 1 else "Not Spam"