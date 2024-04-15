import sys

from llama_index.core import SimpleDirectoryReader
from nltk import WordNetLemmatizer
from preprocessing.pre_processing import pdf_preprocessor
from chroma_vector_db import chroma_db
from vectorizors.llama_vectorizor import LlamaEmbeder
from threading import Thread
import time

thread_running = True

"""Driver function to create ephemeral chromadb vector database useful for prototyping and running locally"""
def my_forever_while():
    """
    Continues to run program and call the input function
    :return:
    """
    global thread_running

    start_time = time.time()

    # run this while there is no input
    while thread_running:
        time.sleep(0.1)

        if time.time() - start_time >= 5:
            start_time = time.time()
            print('Waiting for input')

        take_input()


def take_input():
    """
    User input function call model and return the top 10 documents that match a query. The number of docs returned can
    be tuned by passing a parameter to query currently using default
    :return:
    """
    print('To exit program: q!\n')
    user_input = input('Type query: ')

    if user_input == 'q!':
        sys.exit()
    """the corpus is very small and the model is not being tuned on the corpus 
    I dont expect that the results would improve due to the limited documents but 
    it might be useful given a larger corpus"""
    sample_vector = llama_model.embedding(user_input)
    results = query_engine.query(query_embeddings=sample_vector)

    print(f'Top documents given query {user_input}')
    for res in results.get('metadatas')[0]:
        print(res.get('file_name'))


if __name__ == '__main__':
    """The project is intended to be an extremely MVP for a RAG pipeline I originally intended to flesh out different 
    parts and perform testing to compare the performance of different preprocessing schemas different vector 
    calculators ,etc but that was very ambitious given the intended amount of time the assignment is intended to 
    take. It is also very inefficient since it doesnt persist the db and instead of reads in and calculates the 
    vectors for the documents for the same reason. **I worked on this during the weekend so I didn't have an 
    opportunity to clarify step 2 <broken into text chunks based on either a fixed length or paragraph.> Is the 
    intent to create embedings for the entire article or to preprocess the pdf in a way to generate the most useful 
    representation for querying. I understand that there are ways of combing word vector to represent sentences so it 
    might be possible to combine different paragraph vectors to represent a document. It might also be possible to 
    keep the paragraph vectors separate and implement a voting system ie return top M count the number of times each 
    paragraph corresponds to and article and return the top N. Either way I decided for the sake of expediency to only
    calc vectors for the text what was read in by default byt the llama doc reader, 
    since text that was ingested always included the abstract 
    ie a summary of the entire journal and thus a good representation of the entire doc"""

    t1 = Thread(target=my_forever_while)
    t2 = Thread(target=take_input)
    path = 'local docs/'
    pdfs = SimpleDirectoryReader(path).load_data()
    word_net_normalizer = WordNetLemmatizer()
    llama_model = LlamaEmbeder()
    processed_texts = pdf_preprocessor(pdfs=pdfs, word_normalizer=word_net_normalizer)
    query_engine = chroma_db(documents=processed_texts, model=llama_model)
    t1.start()
    t2.start()

