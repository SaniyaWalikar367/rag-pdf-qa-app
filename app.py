import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


# Page config
st.set_page_config(page_title="RAG PDF QA App", layout="wide")

st.title(
    "📄 Document Question Answering System using RAG "
    "(LangChain + Streamlit)."
)

# Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = " WRITE YOUR HUGGING FACE TOKEN"

# Multi PDF uploader
uploaded_files = st.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

pdf_paths = []

if uploaded_files and len(uploaded_files) > 0:
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            pdf_paths.append(tmp_file.name)

    source_msg = f"Uploaded {len(pdf_paths)} PDF(s)"
else:
    pdf_paths = ["data/paracetamol.pdf"]
    source_msg = "Default PDF (paracetamol.pdf)"



# Build RAG
@st.cache_resource
def build_rag_system(pdf_paths):

    documents = []

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    hf_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipe)

    return db, llm, documents, chunks

# Build once
with st.spinner("Building RAG pipeline (loading PDF(s), chunking, embeddings)..."):
    db, llm, documents, chunks = build_rag_system(tuple(pdf_paths))

st.success(
    f"{source_msg} loaded successfully : {len(documents)} pages | {len(chunks)} chunks"
)

# Conversation memory

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Question box
question = st.text_input("Ask your question from the document(s):")

if st.button("Get Answer"):

    if question.strip() == "":
        st.warning("Please enter a question.")
    else:

        with st.spinner("Retrieving relevant chunks and generating answer..."):

            results = db.similarity_search_with_score(question, k=3)

            docs = [r[0] for r in results]
            scores = [r[1] for r in results]

            context = "\n\n".join([d.page_content for d in docs])

            # ---- use conversation memory ----
            history_text = ""
            for h in st.session_state.chat_history[-3:]:
                history_text += (
                    f"Question: {h['question']}\n"
                    f"Answer: {h['answer']}\n\n"
                )

            prompt = f"""
Use the conversation history and the context to answer the question.

Conversation history:
{history_text}

Context:
{context}

Current question:
{question}
"""

            answer = llm.invoke(prompt)

        # store in memory
        st.session_state.chat_history.append(
            {"question": question, "answer": answer}
        )

        # Final Answer
        st.subheader(" Final Answer")
        st.write(answer)

        # Retrieved Chunks + Scores

        st.subheader("Top-3 Retrieved Chunks (with similarity scores)")

        for i, (doc, score) in enumerate(zip(docs, scores), start=1):
            with st.expander(f"Chunk {i} | Similarity score : {score:.4f}"):
                st.write(doc.page_content)

        # Show conversation history

        st.subheader("Conversation history")

        for i, h in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"**Q{i}:** {h['question']}")
            st.markdown(f"**A{i}:** {h['answer']}")