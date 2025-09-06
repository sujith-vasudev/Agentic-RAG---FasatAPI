import os

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from app.config import settings



def load_vector():
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    # will load the indexed file from external source later
    vectorstore = FAISS.load_local(
        settings.VECTORSTORE_INDEX_PATH,
        embeddings=OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_type="similarity")

__all__ = ["load_vector"]