from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from . import load
import pandas as pd
import unicodedata
import string
import re


def get_text_from_soup(soup_object, text_tag='span'):
    text_tags = [tag for tag in soup_object.find_all(text_tag) if tag.parent.name != 'td']
    text_tags = [tag for tag in text_tags if len(tag.text) > 1]
    text_tags = [unicodedata.normalize("NFKD", tag.text.strip()) for tag in text_tags]
    full_text = " ".join(text_tags)

    return full_text


def split_into_sentences(text, min_words=5):
    sentences = sent_tokenize(text)
    sentences = [sentence for sentence in sentences if len(sentence) > min_words]

    return sentences


def lemmatize_sentences_and_remove_stop_words(list_of_sentences, additional_stop_words=[], remove_digits=False):
    lemmatizer = WordNetLemmatizer()
    custom_stop_words = set(stopwords.words('english') + additional_stop_words)

    cleaned_sentences = list()

    for sentence in list_of_sentences:
        if remove_digits:
            sentence = re.sub(r"\d(\.)\d", " ", sentence)
            sentence = re.sub(r"(\d)+", "", sentence)
        dirty_tokens = sentence.split(" ")
        cleaned_tokens = [ lemmatizer.lemmatize(word.lower().strip()) for word in dirty_tokens if len(word) > 1]
        cleaned_tokens = [ word for word in cleaned_tokens if word not in custom_stop_words ]
        cleaned_sentence = " ".join(cleaned_tokens)
        
        exclude_punctuation = string.punctuation
        exclude_table = str.maketrans("", "", exclude_punctuation)
        cleaned_sentence = cleaned_sentence.translate(exclude_table)
        
        cleaned_sentences.append(cleaned_sentence)

    return cleaned_sentences


def turn_filing_into_sentences_df(report_link):
    """ 
    Purpose: Turn an SEC filing into a dataframe of dirty and clean sentences.
    """

    entry_soup = load.load_soup_from_file_or_edgar(report_link)
    full_text = get_text_from_soup(entry_soup)
    sentences = split_into_sentences(full_text)
    cleaned_sentences = lemmatize_sentences_and_remove_stop_words(sentences, remove_digits=True)
    report_df = pd.DataFrame( zip(sentences, cleaned_sentences), columns=["Dirty", "Clean"])
    
    return report_df