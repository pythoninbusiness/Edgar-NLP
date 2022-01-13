from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from . import load
import pandas as pd
import unicodedata
import string
import re


def remove_digits(list_of_documents):
    list_of_docs_without_digits = list()
    for doc in list_of_documents:
        list_of_docs_without_digits.append(re.sub(r"\d+", " ", doc))
    return list_of_docs_without_digits


def lowercase(list_of_documents):
    if type(list_of_documents) == pd.Series:
        return list_of_documents.str.lower()
    else:
        return [doc.lower() for doc in list_of_documents]


def remove_punctuation(list_of_documents):
    list_of_docs_without_punct = list()
    for doc in list_of_documents:
        list_of_docs_without_punct.append(
            re.sub(r'[%s]' % re.escape(string.punctuation + "’"), ' ', doc)
        )
    return list_of_docs_without_punct


def remove_extra_spaces(list_of_documents):
    list_of_clean_docs = list()
    for doc in list_of_documents:
        clean_doc = re.sub(r"\n", " ", doc)
        list_of_clean_docs.append(
            re.sub(r"\s{2,}", " ", clean_doc)
        )
    return list_of_clean_docs


def lemmatize(list_of_documents):
    list_of_cleaned_documents = list()
    lemmatizer = WordNetLemmatizer()
    for doc in list_of_documents:
        dirty_tokens = doc.split(" ")
        cleaned_tokens = [ lemmatizer.lemmatize(word) for word in dirty_tokens if len(word) > 1]
        cleaned_sentence = " ".join(cleaned_tokens)
        list_of_cleaned_documents.append(cleaned_sentence)
    return list_of_cleaned_documents


def stop_word_removal(list_of_documents):
    list_of_cleaned_documents = list()
    stop_words = stopwords.words('english')
    for doc in list_of_documents:
        dirty_tokens = doc.split(" ")
        cleaned_sentence = " ".join( [token for token in dirty_tokens if token not in stop_words] )
        list_of_cleaned_documents.append(cleaned_sentence)
    
    return list_of_cleaned_documents


def typical_preprocess(list_of_documents):
    list_of_clean_docs = remove_digits(list_of_documents)
    list_of_clean_docs = lowercase(list_of_clean_docs)
    list_of_clean_docs = remove_punctuation(list_of_clean_docs)
    list_of_clean_docs = remove_extra_spaces(list_of_clean_docs)
    list_of_clean_docs = lemmatize(list_of_clean_docs)
    list_of_clean_docs = stop_word_removal(list_of_clean_docs)
    list_of_clean_docs = [doc.strip() for doc in list_of_clean_docs]
    return list_of_clean_docs


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


def turn_filing_into_sentences_df(report_link):
    """ 
    Purpose: Turn an SEC filing into a dataframe of dirty and clean sentences.
    """

    entry_soup = load.load_soup_from_file_or_edgar(report_link)
    full_text = get_text_from_soup(entry_soup)
    sentences = split_into_sentences(full_text)
    cleaned_sentences = typical_preprocess(sentences)
    report_df = pd.DataFrame( zip(sentences, cleaned_sentences), columns=["Dirty", "Clean"])
    
    return report_df
