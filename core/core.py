from googletrans import Translator


def start_pipeline(clf, tfidf, text, translate=True):
    if translate:
        text = _translate_text(text)
    x = _tfidf_transform(tfidf, text)
    return _prediction_text(clf, x)


def _prediction_text(clf, text_tfdf):
    return clf.predict(text_tfdf)


def _tfidf_transform(tfidf_, text):
    return tfidf_.transform(text)


def _translate_text(text, dest='en', tokernize='sentences'):
    translator = Translator()
    return translator.translate(text, dest=dest).text


def _tokenizer_text(text, to='sentences'):
    return text