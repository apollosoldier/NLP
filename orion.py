import numpy as np
import spacy
import gensim
import torch
from transformers import AutoTokenizer, AutoModel

class TextModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

    def preprocess_text(self, text):
        """
        Preprocess the text by lowercasing, removing punctuation and white spaces, and lemmatizing
        """
        doc = self.nlp(text)
        return [token.lemma_ for token in doc if not (token.is_punct or token.is_space or token.is_stop)]

    def lda_model(self, texts, num_topics=10, num_passes=10, num_words=10):
        """
        Train an LDA model on the preprocessed texts and return the most probable words in each topic
        """
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        dictionary = gensim.corpora.Dictionary(preprocessed_texts)
        bow_corpus = [dictionary.doc2bow(text) for text in preprocessed_texts]
        lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=num_passes)
        return [sorted(lda_model.show_topic(topicid, topn=num_words), key=lambda x: -x[1]) for topicid in range(num_topics)]

    def bert_representation(self, texts):
        """
        Encode the texts using BERT and return the resulting representations
        """
        input_ids = torch.tensor([self.tokenizer.encode(text, return_tensors="pt") for text in texts]).squeeze(1)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
        return last_hidden_states.mean(dim=1).numpy()
if __name__ == "__main__":
    texts = [
        "This is a sentence about natural language processing.",
        "Another sentence about NLP and its applications.",
        "Yet another sentence that talks about the field of NLP."
    ]
    tm = TextModel()

    preprocessed_texts = tm.preprocess_texts(texts)

    ngrams = tm.create_ngrams(preprocessed_texts)

    tm.train_LDA(ngrams)

    topic_distributions = tm.get_topic_distributions(ngrams)

    encoded_texts = tm.encode_texts(texts)

    cosine_similarities = tm.get_cosine_similarities(encoded_texts)

    print("N-grams:")
    print(ngrams)
    print("\nTopic Distributions:")
    print(topic_distributions)
    print("\nEncoded Texts:")
    print(encoded_texts)
    print("\nCosine Similarities:")
    print(cosine_similarities)
