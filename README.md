# Multi-Document PDF Question Answering System using RAG
--Project Overview--
This project implements a Retrieval-Augmented Generation (RAG) based document question answering systemthat allows users to upload one or more PDF documents and ask natural language questions.
The system retrieves the most relevant text chunks from the uploaded documents using vector similarity search and generates answers strictly based on the retrieved content.
The application is implemented using LangChain for building the document processing and retrieval pipeline, FAISS for efficient vector indexing and similarity search, transformer models from Hugging Face, and a web-based user interface developed using Streamlit.
The objective of this project is to build a reliable, transparent and interactive system for querying information from documents while reducing hallucinations by grounding the language model with retrieved evidence.

# Key Features
-Upload and process multiple PDF documents
-Automatic document loading and preprocessing
-Text chunking with overlapping segments
-Embedding generation for all chunks
-Vector-based retrieval using FAISS
-Top-3 relevant chunk retrieval
-Display of similarity score for each retrieved chunk
-Answer generation only from retrieved context
-Conversation memory to support follow-up questions
-Interactive and user-friendly web interface

# System Architecture
The system follows a standard RAG pipeline.
Uploaded PDF documents are loaded and merged, then split into overlapping text chunks.
Each chunk is converted into an embedding and stored in a FAISS vector database.
When the user asks a question, the query is embedded and the top-3 most relevant chunks are retrieved.
The retrieved chunks and recent conversation history are combined into a prompt and provided to a language model for answer generation.
Finally, the answer, retrieved chunks and their similarity scores are displayed in the user interface.

# Methodology
1.Document Loading:
All uploaded PDF documents are loaded and merged into a unified document list.

2.Text Chunking:
Documents are split using a recursive character splitter with a chunk size of 500 characters and an overlap of 100 characters.

3.Embedding Generation:
Each chunk is converted into a dense vector representation using a sentence-embedding model.

4.Vector Store Creation:
All embeddings are indexed in a FAISS vector database.

5.Retrieval:
For every user query, the top-3 most similar chunks are retrieved along with their similarity scores.

6.Context Construction:
The retrieved chunks are concatenated into a single context block.

7.Conversation Memory:
Recent question–answer pairs are stored and injected into the prompt to support contextual follow-up queries.

8.Answer Generation:
A text-to-text language model generates the final answer using only the retrieved context and conversation history

# User Interface
The Streamlit interface provides:
-multi-document file upload,
-question input box,
-loading spinner during processing,
-display of the generated answer,
-expandable view of retrieved chunks,
-similarity score for each chunk,
-and conversation history.

# Tools and Technologies
-Python
-Langchain
-FAISS
-Hugging Face Transformers
-Streamlit

# Conclusion
The developed system successfully demonstrates a multi-document RAG-based question answering pipeline with transparent retrieval and grounded answer generation.By combining semantic search with large language models and maintaining conversational memory, the application enables effective and reliable exploration of document collections.The system is suitable for use in academic document analysis, technical documentation search and knowledge extraction tasks.

## Author 
SANIYA WALIKAR
