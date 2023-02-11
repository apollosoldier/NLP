import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

class SonotoriDesu:
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        self.pipeline = None

    def train_bert(self, model_type='bert-base-cased', max_length=512, batch_size=32):
        self.pipeline = Pipeline([
            ('bert', transformers.BertTokenizer.from_pretrained(model_type, do_lower_case=False)),
            ('clf', LinearSVC(C=1))
        ])

        self.pipeline.fit(self.texts, self.labels)

    def predict_bert(self, texts):
        return self.pipeline.predict(texts)

    def evaluate_bert(self, texts, labels):
        predictions = self.pipeline.predict(texts)
        print(confusion_matrix(labels, predictions))
        print(classification_report(labels, predictions))

    def train_tfidf(self, ngram_range=(1,2), min_df=5, max_df=0.8):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df)),
            ('clf', LinearSVC(C=1))
        ])

        self.pipeline.fit(self.texts, self.labels)

    def predict_tfidf(self, texts):
        return self.pipeline.predict(texts)

    def evaluate_tfidf(self, texts, labels):
        predictions = self.pipeline.predict(texts)
        print(confusion_matrix(labels, predictions))
        print(classification_report(labels, predictions))
        
if __name__ == "__main__":
    # Load the data
    texts = ["example text 1", "example text 2", "example text 3"]
    
    # Initialize the TextModel class
    text_model = SonotoriDesu(texts)
    
    # Choose the type of model you want to use
    model_type = "LDA"
    
    # Train the model
    text_model.train_model(model_type)
    
    # Get the topic model
    topic_model = text_model.get_topic_model()
    
    # Print the topics
    print("Topics:")
    for idx, topic in enumerate(topic_model.print_topics()):
        print("Topic #{}: {}".format(idx, topic))
