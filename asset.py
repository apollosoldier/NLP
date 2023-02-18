import lux
import numpy as np
import pandas as pd
import os
import seaborn as sns

#Visualization tools
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#NLP tools
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import gensim.downloader as api
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
from typing import TypeVar, Callable, List, Union

import re
import pandas as pd
from collections import Counter
import multiprocessing
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from typing import List
from gensim.models import Phrases
from sklearn.feature_extraction import text
from gensim.utils import simple_preprocess
from spacy.parts_of_speech import IDS as POS_map
import spacy
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

from tqdm import tqdm_notebook as tqdm
T = TypeVar('T', bound=Union[List[str], List[List[str]]])

class DataPreparation:
    def __init__(self, data_frame_path: str = "News_Category_Dataset_v3.json"):
        nltk.download("stopwords", quiet=True)
        self.dataset = pd.read_json(
            data_frame_path, lines=True, dtype={"headline": str}
        )

        self.stopwords = None
        self.lemmatized = None
        self.source_text_filtered = None
        self.source_text_words = None
        self.texts_stop_words_removed = None
        self.n_gram_texts = None


    def remove_duplicate(self) -> None:
        dup = dataset.duplicated().sum()
        print(f"The dataset contains {dup} duplicates which can be easliy dropped")
        self.dataset = self.dataset.drop_duplicates()
        print(f"The dataset contains {self.dataset.duplicated().sum()} duplicates.")

    def make_data_for_plot(self):
        df = self.dataset.copy()
        news_df = df[df["date"] >= pd.Timestamp(2012, 1, 1)]

        cat = news_df.category.value_counts(normalize=True, sort=False) * 1000
        df_counts = cat.rename_axis("Unique_cat").to_frame("counts")
        df_counts = df_counts.reset_index()
        return news_df, cat, df_counts

    def plot_population(self, top_n):
        df = self.dataset.copy()
        new_df = df.drop(columns=["authors", "link", "date"])
        cat_df = pd.DataFrame(new_df["category"].value_counts()).reset_index()
        cat_df.rename(
            columns={"index": "news_classes", "category": "numcat"}, inplace=True
        )

        # Visualize top 10 categories and proportion of each categories in dataset
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(
            np.array(cat_df.news_classes)[:top_n], np.array(cat_df.numcat)[:top_n]
        )
        for p in ax.patches:
            ax.annotate(p.get_height(), (p.get_x() + 0.01, p.get_height() + 50))
        plt.title(f"TOP {top_n} Categories of News articles", size=15)
        plt.xlabel("Categories of articles", size=14)
        plt.xticks(rotation=45)
        plt.ylabel("Number of articles", size=14)
        plt.show()

    def split_text(self, text):
        return text.split(" ")

    def split_words(self, texts: List[str]) -> List[List[str]]:
        words: List[List[str]]
        with multiprocessing.Pool() as pool:
            words = [
                result for text, result in zip(texts, pool.imap(split_text, texts))
            ]
        return words

    def compute_word_occurrences(self, texts: List[str]) -> pd.DataFrame:
        from collections import Counter

        words = [word for sublist in texts for word in sublist]
        word_count = Counter(words)
        return pd.DataFrame(
            {"Word": list(word_count.keys()), "Count": list(word_count.values())}
        )

    def check_data_quality(self, texts: List[str]) -> bool:
        if not all(map(lambda x: isinstance(x, str), texts)):
            raise ValueError("Input data contains something different than strings.")
        if not all(map(lambda x: x != np.nan, texts)):
            raise ValueError("Input data contains NaN values.")
        return True

    def force_format(self, texts):
        return [str(text) for text in texts]

    def filter_text(self, texts_in):
        """Removes incorrect patterns from a list of texts, such as hyperlinks, bullet points and so on"""

        # Define the regular expressions
        pattern_url = re.compile(r"https?:\/\/[A-Za-z0-9_.-~\-]*")
        pattern_special_chars = re.compile(r"[(){}\[\]<>]")
        pattern_entities = re.compile(r"&amp;#.*;|&gt;|â€™|&#x200B;|-")
        pattern_whitespaces = re.compile(r"\s+")
        pattern_email = re.compile(r"[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+\.[a-zA-Z0-9-_.]+")
        pattern_phone = re.compile(r"\(?\d{3}\D{0,3}\d{3}\D{0,3}\d{4}")
        pattern_twitter = re.compile(r"@\S+( |\n)")
        pattern_star = re.compile(r"\*")

        # Remove the patterns from the text
        texts_out = pattern_url.sub(" ", texts_in)
        texts_out = pattern_special_chars.sub(" ", texts_out)
        texts_out = pattern_entities.sub(" ", texts_out)
        texts_out = pattern_whitespaces.sub(" ", texts_out)
        texts_out = pattern_email.sub("", texts_out)
        texts_out = pattern_phone.sub("", texts_out)
        texts_out = pattern_twitter.sub("", texts_out)
        texts_out = pattern_star.sub("", texts_out)

        return texts_out

    def make_my_pipeline(
        self,
        value: T,
        function_pipeline: List[Callable[[T], T]],
    ) -> T:
        """A generic Unix-like pipeline

        :param value: the value you want to pass through a pipeline
        :param function_pipeline: an ordered list of functions that
            comprise your pipeline
        """
        return reduce(lambda v, f: f(v), function_pipeline, value)

    def sent_to_words(self, sentences):
        return map(
            lambda sentence: simple_preprocess(str(sentence), deacc=True), sentences
        )

    def remove_stop_words(self, texts):
        stop_words = stopwords.words("english")
        return list(
            map(lambda txt: [word for word in txt if word not in stop_words], texts)
        )

    def get_stopwords(self, additional_stopwords=None):
        with open("stopwords.txt") as f:
            stopwords = [line.strip() for line in f]
        stopwords = text.ENGLISH_STOP_WORDS.union(stopwords)

        if additional_stopwords:
            stopwords = stopwords.union(additional_stopwords)
        stopwords = sorted(list(stopwords), key=str.lower)

        return stopwords
   
    def lemmatize_texts(
        self,
        texts,
        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN', 'INTJ', 'NUM', 'X'],
        forbidden_postags=[],
        as_sentence=False,
        get_postags=False,
        spacy_model=None,
    ):
        """Lemmatize a list of texts.

            Please refer to https://spacy.io/api/annotation for details on the allowed
        POS tags.
        @params:
            - texts_in: a list of texts, where each texts is a string
            - allowed_postags: a list of part of speech tags, in the spacy fashion
            - as_sentence: a boolean indicating whether the output should be a list of sentences instead of a list of word lists
        @return:
            - A list of texts where each entry is a list of words list or a list of sentences
        """
        texts_out = []

        if allowed_postags and forbidden_postags:
            raise ValueError("Can't specify both allowed and forbidden postags")
        if forbidden_postags:
            allowed_postags = list(set(POS_map.keys()).difference(set(forbidden_postags)))
        if not spacy_model:
            print("Loading spacy model")
            spacy_model = spacy.load("en_core_web_lg")
        print("Beginning lemmatization process")
        total_steps = len(texts)

        docs = spacy_model.pipe(texts)

        for i, doc in tqdm(enumerate(docs), total=total_steps):
            if get_postags:
                texts_out.append(
                    [
                        "_".join([token.lemma_, token.pos_])
                        for token in doc
                        if token.pos_ in allowed_postags
                    ]
                )
            else:
                texts_out.append(
                    [token.lemma_ for token in doc if token.pos_ in allowed_postags]
                )
        if as_sentence:
            texts_out = [" ".join(text) for text in texts_out]
        return texts_out


    def build_dataset(self, n_gram):
        try:
            source = self.force_format(self.dataset["headline"])
            print(f"[+] Checked data quality : {self.check_data_quality(source)}")
            source_text_filtered = list(map(self.filter_text, source))
            source_text_words = list(self.sent_to_words(source_text_filtered))
            print("[+] Built data for data preparation tool")
            texts_stop_words_removed = self.remove_stop_words(source_text_words)
            print("[+] Removed stop words")
            n_gram_texts = self.create_ngrams(
                texts_stop_words_removed,
                n=n_gram,
                min_count=15,
                threshold=10,
                convert_sent_to_words=False,
                as_str=True,
            )
            print(f"[+] Created n_gram = [{n_gram}]")
            n_gram_texts_lemmatized = self.lemmatize_texts(texts = n_gram_texts)
            """output_strings = pipeline(
                            texts_worded,
                            [remove_stop_words, create_ngrams, lemmatize_texts, compute_word_occurrences],
                        )"""
            print("[+] Lemmatized")
            print("[+] Operation successfully completed")
            self.source_text_filtered = source_text_filtered
            self.source_text_words = source_text_words
            self.texts_stop_words_removed = texts_stop_words_removed
            self.n_gram_texts = n_gram_texts
        except Exception as e:
            raise Exception("Error while building the dataset: {}".format(str(e)))
        finally:
            self.lemmatized = n_gram_texts_lemmatized
        return n_gram_texts_lemmatized

        
    def create_ngrams(
        self,
        texts,
        n=2,
        min_count=15,
        threshold=10,
        convert_sent_to_words=False,
        as_str=True,
    ):
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
        return map(
            lambda sentence: simple_preprocess(str(sentence), deacc=True), sentences
        )

    def scrappe_link(self, url):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            }
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, "html.parser")

            header = soup.find("h1", {"class": "headline"})
            if header:
                title = header.text
            else:
                title = None
            paragraphs = [p.text for p in soup.find_all("p")]
            article = " ".join(paragraphs)
            article = self.filter_text(article)
            return {"link": url, "title": title, "article": article}
        except Exception as e:
            return {"link": url, "title": np.nan, "article": np.nan}

    def scrape_links(self, links):
        articles = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_link = {
                executor.submit(self.scrappe_link, link): link for link in links
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_link)):
                link = future_to_link[future]
                
    def circular_bar(self, col_name, col_counts):
        news_df, cat, df = self.make_data_for_plot()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot the circular bar chart
        ax1.set_title("Circular Bar Chart")
        ax1.set_axis_off()
        ax1 = plt.subplot(121, polar=True)
        upperLimit = 100
        lowerLimit = 40

        maxi = df[col_counts].max()
        slope = (maxi - lowerLimit) / maxi
        heights = slope * df[col_counts] + lowerLimit

        width = 2 * np.pi / len(df.index)

        indexes = list(range(1, len(df.index) + 1))
        angles = [element * width for element in indexes]
        angles
        bars = ax1.bar(
            x=angles,
            height=heights,
            width=width,
            bottom=lowerLimit,
            linewidth=1,
            edgecolor="black",
            color="#61a4b2",
        )

        labelPadding = 4

        for bar, angle, height, label in zip(bars, angles, heights, df[col_name]):
            rotation = np.rad2deg(angle)

            alignment = ""
            if angle >= np.pi / 2 and angle < 3 * np.pi / 2:
                alignment = "right"
                rotation = rotation + 180
            else:
                alignment = "left"
            ax1.text(
                x=angle,
                y=lowerLimit + bar.get_height() + labelPadding,
                s=label,
                ha=alignment,
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
            )
        # Plot the pie chart of top 20 categories of news articles
        df = self.dataset.copy()
        new_df = df.drop(columns=["authors", "link", "date"])
        cat_df = pd.DataFrame(new_df["category"].value_counts()).reset_index()
        cat_df.rename(columns={"index": "news_classes", "category": "numcat"}, inplace=True)

        ax2.set_title("Pie Chart of Top 20 categories of news articles")
        A = ax2.pie(
            cat_df["numcat"][:20],
            labels=cat_df["news_classes"][:20],
            autopct="%1.1f%%",
            startangle=90,
            labeldistance=1.08,
            pctdistance=1.03,
            rotatelabels=45,
        )
        plt.tight_layout()
        plt.show()


    def plot_circular(self):
        self.circular_bar("Unique_cat", "counts")
        plt.show()
