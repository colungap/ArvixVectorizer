import sys
import chromadb
from vectorizors.llama_vectorizor import LlamaEmbeder

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


def chroma_db(documents: [str], model: LlamaEmbeder):
    """
    chroma collection creation util function. There are several ways to intialize and populate the db but this was the
    only way I found that is done entirely on the local machine avoiding api calls. Not super familiar with the llama
    API so it may be possible to avoid having to parse out the documents and create embeddings for them prior to adding
    to db. The API method is much cleaner and the Document obj likely to be better than the document format I'm using,
    although I tried to maintain the import parts specifically the original id and the metadata.
    It is also possible to use built-in embedding functions included in Chroma, but I wanted an implementation where the
    database and the vector calculations where not tightly coupled. In theory, you can use whatever vector calc function
    as long it returns [Embedding/np.ndarray] or Embedding/np.ndarray.
    :param documents:
    :param model:
    :return:
    """
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("ArvixChroma")
    texts = [doc.text for doc in documents]
    emb = model.batch_embedding(texts)
    ids = [doc.doc_id for doc in documents]
    meta = [doc.metadata for doc in documents]
    chroma_collection.add(embeddings=emb, ids=ids, metadatas=meta)

    return chroma_collection
