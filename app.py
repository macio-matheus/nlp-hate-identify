import joblib
from flask import Flask
from flask import jsonify, request

from core.core import start_pipeline

app = Flask(__name__)

cls = joblib.load('./data/model_mlp_cls.pkl')
tfidf = joblib.load('./data/tfidf_vectorizer.pkl')


@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'hello': 'world'})


@app.route('/services/classify_text', methods=['POST'])
def classify_text():

    text = request.get_json()['text']
    res = start_pipeline(cls, tfidf, text)
    return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True, port=6543, host='0.0.0.0')
