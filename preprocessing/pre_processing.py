import string
from functools import partial
from typing import List

from llama_index.core import Document
from nltk import StemmerI
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def _parse(normalizer, stop_words: {str}, page: Document) -> Document:
    """Parsing function to remove stop words and perform basic preprocessing, while there aren't other functions the
    idea is that other parser could take different inputs for the different steps in preprocessing docs
    to fit the downstream use case currently only supports different kinds of steming/lemming. This would make it simple
    to experiment with different types of stemmers/lemmers/stopwords etc"""
    if isinstance(normalizer, WordNetLemmatizer):
        normalizer = partial(normalizer.lemmatize)

    elif isinstance(normalizer, StemmerI):
        normalizer = partial(normalizer.stem)

    else:
        normalizer = partial(lambda x: x)

    processed_txt = [normalizer(token).lower().translate(str.maketrans('', '', string.punctuation))
                     for token in word_tokenize(page.text)
                     if token not in stop_words and not token.isdigit()]

    page.text = " ".join(processed_txt)
    return page


def chuncker(doc: [str], size: int = None):
    """Chunking function to break up a document into multiple pieces, doc embedding is limited and this allows to
    create vectors for the entire doc. Not currently being used.
    The llama doc reader is only reading in the first page"""
    start = 0
    end = len(doc)
    if size is None:
        size = end
    chunked_doc = []
    for curr in range(start, end, size):
        chunked_doc.append(" ".join(doc[curr:curr + size]))

    return chunked_doc


def pdf_preprocessor(pdfs: List[Document], word_normalizer, stop_words=None, chunk_size: int = None) -> [str]:
    """
    utility function to iterate and preprocess a group of Documents.
    :param pdfs:
    :param word_normalizer:
    :param stop_words:
    :param chunk_size:
    :return:
    """
    if stop_words is None:
        stop_words = set(stopwords.words("english"))

    all_text = [_parse(normalizer=word_normalizer, stop_words=stop_words, page=pdf) for pdf in pdfs]

    return all_text
