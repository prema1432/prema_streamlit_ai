from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain_openai import OpenAIEmbeddings

CHROMA_DB_DIRECTORY = "chroma_db"
CHAT_OPENAI_MODEL_NAME = "gpt-3.5-turbo"
# CHAT_OPENAI_MODEL_NAME = "gpt-4-turbo"
EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 20

GLOBAL_EMBEDDINGS = SentenceTransformerEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
# GLOBAL_EMBEDDINGS = OpenAIEmbeddings()