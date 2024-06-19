import streamlit as st

st.set_page_config(page_title="Premanath", page_icon="👋", layout="centered", initial_sidebar_state="auto",
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
