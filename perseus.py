import spacy
from sklearn.feature_extraction.text import CountVectorizer

POS_map = {'ADJ': 'a', 'ADP': 'r', 'ADV': 'r', 'AUX': 'v', 'CONJ': 'c', 'CCONJ': 'c',
           'DET': 'd', 'INTJ': 'u', 'NOUN': 'n', 'NUM': 'm', 'PART': 'p', 'PRON': 'o',
           'PROPN': 'n', 'PUNCT': '', 'SCONJ': 'c', 'SYM': '', 'VERB': 'v', 'X': ''}

class TextClassifier:
    def __init__(self, vectorizer=None, classifier=None):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def fit(self, X, y):
        if not self.vectorizer:
            self.vectorizer = CountVectorizer()

        X = self.vectorizer.fit_transform(X)
        if not self.classifier:
            from sklearn.naive_bayes import MultinomialNB
            self.classifier = MultinomialNB()

        self.classifier.fit(X, y)

    def predict(self, X):
        X = self.vectorizer.transform(X)
        return self.classifier.predict(X)
    
    @staticmethod
    def lemmatize_texts(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'], forbidden_postags=[], as_sentence=False, get_postags=False, spacy_model=None):
        if allowed_postags and forbidden_postags:
            raise ValueError("Can't specify both allowed and forbidden postags")

        if forbidden_postags:
            allowed_postags = list(set(POS_map.keys()).difference(set(forbidden_postags)))

        if not spacy_model:
            spacy_model = spacy.load('en_core_web_md')

        lemmatized_texts = []
        for text in texts:
            doc = spacy_model(text)
            lemmatized_text = [token.lemma_ if not get_postags else f"{token.lemma_}_{token.pos_}" for token in doc if token.pos_ in allowed_postags]
            lemmatized_texts.append(" ".join(lemmatized_text) if as_sentence else lemmatized_text)

        return lemmatized_texts
