import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------- UI ----------------
st.set_page_config(page_title="Telecom AI Copilot", layout="wide")
st.title("📡 Telecom AI Copilot")
st.write("Ask your Telecom related question below.")

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base")

embedding_model = load_embedding_model()
llm = load_llm()

# ---------------- LOAD PDFs ----------------
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

text1 = load_pdf("ZENDS_communication.pdf")
text2 = load_pdf("telecom_ai_Copilot.pdf")
text = text1 + "\n" + text2

# ---------------- SPLIT ----------------
def split_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = split_text(text)

# ---------------- FAISS ----------------
dimension = 384
index = faiss.IndexFlatL2(dimension)

embeddings = embedding_model.encode(chunks)
index.add(np.array(embeddings).astype("float32"))

# ---------------- GENERATE FUNCTION ----------------
def generate_answer(query):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding).astype("float32"), k=2)

    context = ""
    for i in indices[0]:
        context += chunks[i] + "\n"

    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm(prompt, max_length=200)
    return response[0]["generated_text"]

# ---------------- STREAMLIT INPUT ----------------
user_query = st.text_input("Enter your question:")

if user_query:
    answer = generate_answer(user_query)
    st.subheader("Answer:")
    st.write(answer)
