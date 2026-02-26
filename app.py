import streamlit as st
import pandas as pd
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#  PAGE CONFIGURATION:
st.set_page_config(page_title="ZENDS AI Copilot", layout="wide")
st.title(" ZENDS AI Customer Support Copilot")


df = pd.read_csv("reviews.csv")

selected_review = st.selectbox(
    "Select Customer Review",
    df["text"]
)

row = df[df["text"] == selected_review].iloc[0]
intent = row["intent"]
sentiment = row["sentiment"]

st.subheader("📊 Classification Output")
st.write("Intent:", intent)
st.write("Sentiment:", sentiment)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# RAG:
@st.cache_resource
def prepare_rag():
    reader1 = PdfReader("ZENDS_communication.pdf")
    reader2 = PdfReader("telecom_ai_copilot.pdf")

    full_text = ""

    for page in reader1.pages:
        full_text += page.extract_text()

    for page in reader2.pages:
        full_text += page.extract_text()

    # SIMPLE 500 CHAR CHUNKING:
    chunk_size = 400
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]

    embeddings = embedding_model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    return chunks, index


chunks, index = prepare_rag()


#  RAG FUNCTION :
def generate_answer(query):

    query_lower = query.lower()

    query_embedding = embedding_model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"), k=3
    )

    retrieved_chunks = [chunks[i] for i in indices[0]]

    countries = ["india", "usa", "singapore", "thailand"]

    # PRICING EXTRACTION :
    for chunk in retrieved_chunks:

        chunk_lower = chunk.lower()

        # Match product name
        if any(word in chunk_lower for word in query_lower.split()):

            # Match country
            for country in countries:
                if country in query_lower and country in chunk_lower:

                    # Match individual / enterprise
                    if "individual" in query_lower:
                        if "individual" in chunk_lower:
                            return extract_price(chunk, country, "individual")

                    elif "enterprise" in query_lower:
                        if "enterprise" in chunk_lower:
                            return extract_price(chunk, country, "enterprise")

                    else:
                        return extract_price(chunk, country, None)

    # Default fallback
    return retrieved_chunks[0]


def extract_price(chunk, country, user_type):

    parts = chunk.split(",")

    for part in parts:
        part_lower = part.lower()

        if country in part_lower:
            if user_type:
                if user_type in chunk.lower():
                    return f"Official Pricing Detail:\n\n{part.strip()}"
            else:
                return f"Official Pricing Detail:\n\n{part.strip()}"

    return chunk

st.subheader("🤖 AI Generated Response")
if st.button("Generate Response"):
    answer = generate_answer(selected_review)
    st.write(answer)


# Visualization:
st.markdown("---")
st.header("📊 Project Analytics Dashboard")

# Intent Distribution:
st.subheader("Intent Distribution")

intent_counts = df["intent"].value_counts()

st.bar_chart(intent_counts)


# Sentiment distribution:
st.subheader("Sentiment Distribution")

sentiment_counts = df["sentiment"].value_counts()

st.bar_chart(sentiment_counts)


st.subheader("Pricing Coverage Across Countries")

countries = ["USA", "India", "Singapore", "Thailand"]

country_data = {
    "USA": 10,
    "India": 10,
    "Singapore": 10,
    "Thailand": 10
}

country_df = pd.DataFrame.from_dict(country_data, orient="index", columns=["Plans Available"])

st.bar_chart(country_df)


# model summary metrics:
st.subheader("Model Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Reviews", len(df))

with col2:
    st.metric("Intent Classes", df["intent"].nunique())

with col3:
    st.metric("Sentiment Classes", df["sentiment"].nunique())
