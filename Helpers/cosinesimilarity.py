from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from . import textpreprocess


def search_engine(query: str, list_of_options: list, limit=3, add_cols=list()):

    """
    Purpose: Compare a search sentence (string) with a list of options (list of strings). Return the closest match using TF-IDF.

    Example:
    query = "FN6.1-Intangibles Summary-2Q21.xlsx"
    list_of_options = [
        'FN7.1-Intangibles Summary-3Q20 v1.xlsx', 'FN7.2-Intangibles Amortization-3Q20.xlsx', 'FN7.3-Intangibles Future Amort-3Q20 v1.xlsx', 'FN7.4-Goodwill Balance-3Q20.xlsx'
    ]
    search_engine(search_string, list_of_strings)

    output:
    [('FN Intangibles Summary xlsx', 1.0000000000000002),
    ('FN Intangibles Amortization xlsx', 0.48771736377930486),
    ('FN Intangibles Future Amort xlsx', 0.39659910959273864),
    ('FN Goodwill Balance xlsx', 0.2443706761406413)]
    """
    
    cleaned_sentence = textpreprocess.typical_preprocess([query])
    cleaned_options = textpreprocess.typical_preprocess(list_of_options)

    vectorizer = TfidfVectorizer(ngram_range=(1,1)).fit(cleaned_sentence)

    search_vector = vectorizer.transform(cleaned_sentence).toarray()[0]
    option_vectors = vectorizer.transform(cleaned_options).toarray()

    similarity_scores = cosine_similarity(search_vector.reshape(1, -1), option_vectors)[0]

    if len(add_cols) > 0:
        options_and_scores = list(zip(list_of_options, *add_cols, similarity_scores))
        sorted_scores = sorted(options_and_scores, key=lambda tuple: tuple[-1], reverse=True)
    else:
        options_and_scores = list(zip(list_of_options, similarity_scores))
        sorted_scores = sorted(options_and_scores, key=lambda tuple: tuple[1], reverse=True)

    return sorted_scores[:limit]
