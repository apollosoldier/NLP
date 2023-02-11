import re
import pandas as pd
from collections import Counter
import multiprocessing
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import concurrent.futures
from typing import List
from gensim.models import Phrases
from sklearn.feature_extraction import text
from gensim.utils import simple_preprocess
from spacy.parts_of_speech import IDS as POS_map
import spacy

class NLP:

    def __init__(self):
        self.stopwords = None

    def create_ngrams(self, texts, n=2, min_count=15, threshold=10, convert_sent_to_words=False, as_str=True):
        """Identify bigrams or trigrams in texts and return the texts with bigrams or trigrams integrated"""
        if convert_sent_to_words:
            texts = list(self.sent_to_words(texts))

        ngram_model = Phrases(texts, min_count=min_count, threshold=threshold)

        if as_str:
            return [" ".join(ngram_model[t]) for t in texts]

        else:
            return [ngram_model[t] for t in texts]

    def get_stopwords(self, additional_stopwords=None):
        with open("stopwords.txt") as f:
            stopwords = [line.strip() for line in f]

        stopwords = text.ENGLISH_STOP_WORDS.union(stopwords)

        if additional_stopwords:
            stopwords = stopwords.union(additional_stopwords)

        stopwords = sorted(list(stopwords), key=str.lower)

        self.stopwords = stopwords

        return stopwords

    def sent_to_words(self, sentences):
        return map(lambda sentence: simple_preprocess(str(sentence), deacc=True), sentences)

    def scrappe_link(self, url):
        try:
            headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')

            header = soup.find("h1", {"class": "headline"})
            if header:
                title = header.text
            else:
                title = None
            paragraphs = [p.text for p in soup.find_all("p")]
            article = " ".join(paragraphs)
            article = self.filter_text(article)
            return {
                "link": url,
                "title": title,
                "article": article
            }

        except Exception as e:
            return {
                    "link": url,
                    "title": np.nan,
                    "article": np.nan
                }

    def scrape_links(self, links):
        articles = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_link = {executor.submit(self.scrappe_link, link): link for link in links}
            for future in tqdm(concurrent.futures.as_completed(future_to_link)):
                link = future_to_link[future]
    
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
