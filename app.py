from sklearn.externals import joblib
from flask import Flask
from flask import jsonify, request

from core.core import start_pipeline

app = Flask(__name__)

cls = joblib.load('data/model_mlp_cls.pkl')
# cls = joblib.load(open('data/model_random_forest_cls.pkl', 'rb'))
tfidf = joblib.load('data/tfidf_vectorizer.pkl')


@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'hello': 'world'})


@app.route('/services/classify_text', methods=['POST'])
def classify_text():
    text = request.get_json()['text']
    translate = request.get_json()['translate']
    tokenize = request.get_json()['tokenize']
    pred, text_translated = start_pipeline(cls, tfidf, text, translate, tokenize)

    is_hate = False if pred[0] == 0 else True
    res = {'text': text, 'hate': is_hate, 'text_translated': text_translated}

    return jsonify(res)


if __name__ == '__main__':
    app.run(debug=True, port=6543, host='0.0.0.0')
