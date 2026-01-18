import streamlit as st
from langchain.memory import ConversationBufferMemory
import faiss
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 1) Load Vector Store
faiss_index = faiss.read_index("faiss_store/index.faiss")

with open("faiss_store/docs.pkl", "rb") as f:
    docs = pickle.load(f)

# Recreate embeddings object
class LocalEmbeddings:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100)
        doc_texts = [doc.page_content for doc in docs]
        self.vectorizer.fit(doc_texts)
        
    def embed_documents(self, texts):
        return self.vectorizer.transform(texts).toarray().astype(np.float32)
    
    def embed_query(self, text):
        return self.vectorizer.transform([text]).toarray()[0].astype(np.float32)
    
    def similarity_search(self, query, k=4):
        query_embedding = self.embed_query(query)
        query_embedding = np.array([query_embedding])
        distances, indices = faiss_index.search(query_embedding, k)
        return [docs[i] for i in indices[0]]

embeddings = LocalEmbeddings()

# Create retriever
class SimpleRetriever:
    def __init__(self, embeddings, docs):
        self.embeddings = embeddings
        self.docs = docs
        
    def get_relevant_documents(self, query):
        return self.embeddings.similarity_search(query)

retriever = SimpleRetriever(embeddings, docs)

# 2) Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 3) Simple Chatbot Function (without LlamaCpp)
def chat_response(question):
    """Get relevant documents and format response"""
    relevant_docs = retriever.get_relevant_documents(question)
    
    # Combine relevant documents as context
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Simple response generation based on relevant docs
    response = f"Based on the documents, here's what I found:\n\n{context[:500]}..."
    
    return response

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📌 Context-Aware RAG Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
user_input = st.text_input("Ask anything:")

if st.button("Send") and user_input:
    response = chat_response(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": response})

# Display chat
for chat in st.session_state.chat_history:
    st.write("🧑‍💬 You:", chat["user"])
    st.write("🤖 Bot:", chat["bot"])
