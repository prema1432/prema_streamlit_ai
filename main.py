import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from const import CHUNK_SIZE, GLOBAL_EMBEDDINGS, CHUNK_OVERLAP, CHAT_OPENAI_MODEL_NAME
from langchain_openai import ChatOpenAI

st.set_page_config(page_title="Premanath", page_icon=None, layout="centered", initial_sidebar_state="auto",
                   menu_items=None)

st.markdown("""
## Premanath AI: Communicate with your Files 

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Chat GPT. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### Available Features

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need a Google API key for the chatbot to access Google's Generative AI models.

2. **Phone Number**: Enter your phone number to receive a text message with the chatbot's response.

3. **Upload Your Filess**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

4. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

api_key = st.text_input("Enter your Open API Key:", type="password", key="api_key_input")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, api_key):
    vectordb = Chroma.from_documents(documents=text_chunks, embedding=GLOBAL_EMBEDDINGS, persist_directory=f"chroma_db")
    vectordb.persist()


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model_name=CHAT_OPENAI_MODEL_NAME)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, api_key):
    db=Chroma(embedding_function=GLOBAL_EMBEDDINGS, persist_directory=f"chroma_db")
    docs = db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("Prema AI clone chatbot💁")

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Done")

if __name__ == "__main__":
    main()