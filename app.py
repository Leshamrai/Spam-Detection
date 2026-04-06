from flask import Flask, render_template, request
from main import train_and_get_models
from src.preprocessing import preprocess_text
from src.evaluate import evaluate_model

app = Flask(__name__)

# Load models
nb_model, lr_model, tfidf, y_test, y_pred_nb, y_pred_lr = train_and_get_models()

# Calculate metrics
nb_metrics = evaluate_model(y_test, y_pred_nb)
lr_metrics = evaluate_model(y_test, y_pred_lr)


@app.route('/')
def home():
    return render_template('index.html', nb=nb_metrics, lr=lr_metrics)


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    
    processed = preprocess_text(message)
    vector = tfidf.transform([processed]).toarray()
    
    prediction = nb_model.predict(vector)[0]
    
    result = "Spam" if prediction == 1 else "Not Spam"
    
    return render_template(
        'index.html',
        result=result,
        nb=nb_metrics,
        lr=lr_metrics
    )


if __name__ == '__main__':
    app.run(debug=True)