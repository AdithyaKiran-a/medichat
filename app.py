import os
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv, find_dotenv
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["This is a test sentence"])
print(embeddings)

# Load environment variables (make sure GOOGLE_API_KEY is set in your .env)
load_dotenv(find_dotenv())
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def generate_gemini_response(prompt_text):
    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-2.5-flash')
    chat = model.start_chat()
    response = chat.send_message(prompt_text)
    return response.text

def main():
    st.title("Ask Chatbot (Gemini)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask something")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say you don't know.
        Only answer based on the context.

        Context: {context}
        Question: {question}
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Vector store could not be loaded.")
                return

            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            docs = retriever.get_relevant_documents(prompt)
            context = "\n\n".join([doc.page_content for doc in docs])

            formatted_prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context, question=prompt)

            result = generate_gemini_response(formatted_prompt)
            sources = "\n".join([doc.metadata.get('source', 'Unknown source') for doc in docs])

            result_to_show = result + "\n\nSource Docs:\n" + sources

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
