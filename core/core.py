from googletrans import Translator
from nltk.tokenize import sent_tokenize, word_tokenize


def start_pipeline(clf, tfidf, text, translate=True, tokenize=None):
    text_translated = _translate(text, translate, tokenize)
    x = _tfidf_transform(tfidf, text_translated)
    return _prediction_text(clf, x), text_translated


def _translate(text, translate, tokenize):
    if not translate:
        return text

    if tokenize == 'sent':
        text = sent_tokenize(text)
        text = _translate_by_sent_list(text)
    else:
        text = _translate_full_text(text)

    return text


def _translate_full_text(text, dest='en'):
    translator = Translator()
    return translator.translate(text, dest=dest).text


def _translate_by_sent_list(list_tokens, dest='en'):
    translator = Translator()
    text_translated = ''

    for tokens in list_tokens:
        text_translated += translator.translate(tokens, dest=dest).text + ' '

    return text_translated


def _tfidf_transform(tfidf_, text):
    return tfidf_.transform([text])


def _prediction_text(clf, text_tfdf):
    return clf.predict(text_tfdf.toarray())
