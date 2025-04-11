import streamlit as st
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
import tempfile
import os

# Streamlit page configuration
st.set_page_config(page_title="Quick Veda | Doc Analyzer", page_icon="📄", layout="wide")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache embeddings and T5 pipeline
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

@st.cache_resource
def load_t5_pipeline():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        device=0 if torch.cuda.is_available() else -1,
        max_length=500,
        temperature=0.7
    )

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# HTML and CSS (inlined from your documents)
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Quick Veda | Doc Analyzer</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css" />
  <style>
    body {
      margin: 0;
    }
    html {
      -ms-overflow-style: none;
      scrollbar-width: none;
    }
    html::-webkit-scrollbar {
      display: none;
    }
    .navbar-brand img {
      width: 32px;
      height: 32px;
      border-radius: 50%;
    }
    .navbar-brand span {
      font-size: larger;
      font-family: fantasy;
      padding-left: 10px;
    }
    .logo {
      height: 40px;
    }
    li {
      list-style: none;
      display: flex;
      justify-content: space-around;
    }
    .card-img-overlay {
      flex-direction: column;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .card-img {
      height: 500px;
      object-fit: cover;
    }
    .btn.about {
      color: burlywood;
      margin: 10px;
      font-size: 1.5rem;
      border-radius: 10px;
      background-color: #FFFFFF;
    }
    .content {
      width: 100vw;
    }
    .input {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 20px;
    }
    .upload-label {
      font-size: 24px;
      cursor: pointer;
    }
    .upload-btn {
      background-color: #007bff;
      color: white;
      border: none;
      padding: 10px 20px;
      font-size: 16px;
      cursor: pointer;
      border-radius: 5px;
    }
    .upload-btn:hover {
      background-color: #0056b3;
    }
    #uploadStatus {
      margin-top: 20px;
      font-size: 18px;
    }
    #contentArea {
      margin-top: 30px;
    }
    #paper {
      transform: rotate(-44deg);
    }
    .about-container {
      background-image: url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e');
      background-size: cover;
      background-position: center;
      height: 50vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .about-content {
      display: flex;
      flex-direction: row;
      justify-content: center;
      align-items: center;
    }
    .about-text {
      border-radius: 10px;
      background-color: #FFFFFF;
      width: 80vw;
      height: 30vh;
      box-shadow: 0 0 10px #007bff;
      flex: 1;
      display: flex;
      justify-content: center;
      align-items: center;
      text-align: center;
    }
    .about-text div {
      font-size: 24px;
      font-weight: bold;
      color: rgb(223, 143, 45);
    }
  </style>
</head>
<body>
  <nav class="navbar navbar-expand-lg bg-body-tertiary">
    <div class="container-fluid">
      <div class="logo">
        <a class="navbar-brand" href="#">
          <img src="https://media.istockphoto.com/id/183412466/photo/eastern-bluebirds-male-and-female.jpg?s=612x612&w=0&k=20&c=6_EQHnGedwdjM9QTUF2c1ce7cC3XtlxvMPpU5HAouhc=" alt="Logo" />
          <span>Quick Veda</span>
        </a>
      </div>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
      </button>
      <div class="collapse navbar-collapse" id="navbarSupportedContent">
        <ul class="navbar-nav ms-auto mb-2 mb-lg-0">
          <li class="nav-item"><a class="nav-link active" href="#">Home</a></li>
          <li class="nav-item"><a class="nav-link" href="#">Login</a></li>
          <li class="nav-item"><a class="nav-link" href="#">SignUp</a></li>
        </ul>
      </div>
    </div>
  </nav>

  <div class="card text-bg-dark">
    <img src="https://thumbs.dreamstime.com/b/pink-flowers-float-clear-waters-hawaii-soft-white-sand-below-347952870.jpg" class="card-img" alt="..." />
    <div class="card-img-overlay">
      <button class="btn about" onclick="document.getElementById('about').scrollIntoView({behavior: 'smooth'})">About</button>
    </div>
  </div>

  <div class="jumbotron text-center py-5">
    <h1 class="display-4">Upload Your PDFs and Ask Your Questions</h1>
    <p class="lead">Easily upload your PDFs and interact with the content directly. Get answers to your queries.</p>
  </div>
</body>
</html>
"""

# Render HTML header and navbar
st.markdown(html_content, unsafe_allow_html=True)

# Main content
st.markdown('<div class="content container">', unsafe_allow_html=True)

# File uploader
st.markdown("""
<div class="card shadow-sm">
  <div class="card-body">
    <div class="input d-flex flex-column flex-md-row justify-content-center align-items-center gap-3">
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "", accept_multiple_files=False, type=["pdf"], key="pdf_uploader",
    label_visibility="collapsed"
)

question = st.text_input("", placeholder="Enter your question...", key="question_input")
submit_button = st.button("Upload & Analyze", key="upload_button", disabled=not uploaded_file)

st.markdown("""
    </div>
    <div id="uploadStatus" class="text-center mt-3 text-danger fw-bold"></div>
  </div>
</div>
""", unsafe_allow_html=True)

# Content area
st.markdown("""
<div id="contentArea" class="mt-4 p-4 bg-light rounded shadow" style="min-height: 300px;">
  <div class="text-center text-muted" id="placeholderText">
    <i class="fas fa-file-lines fa-2x mb-2"></i>
    <p>No document analyzed yet.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Process uploaded file and question
if submit_button and uploaded_file and question:
    with st.spinner("Processing document..."):
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Load and split document
        try:
            loader = PyPDFLoader(pdf_path)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)

            # Create vector store
            embeddings = load_embeddings()
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)

            # Clean up
            os.unlink(pdf_path)
        except Exception as e:
            st.markdown(f'<div id="contentArea" class="mt-4 p-4 bg-light rounded shadow"><p class="text-danger">Failed to process PDF: {str(e)}</p></div>', unsafe_allow_html=True)
            st.stop()

    with st.spinner("Generating answer..."):
        try:
            # Retrieve relevant chunks
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.invoke(question)
            context = " ".join([doc.page_content for doc in relevant_docs])

            # Run T5 pipeline
            t5_pipeline = load_t5_pipeline()
            prompt = f"""
You are an expert assistant tasked with answering questions based on a document. Provide a detailed, accurate, and comprehensive answer in 5–10 sentences. Use the following context to inform your response, explaining key points clearly. If the context is insufficient, note the limitation and provide the best possible answer based on available information. Do not invent facts.

**Context**: {context}

**Question**: {question}

**Answer**:
"""
            result = t5_pipeline(prompt)[0]["generated_text"]

            # Ensure answer is detailed
            if len(result.split()) < 30:  # If too short
                result += " The document provides limited details on this topic, but the available context suggests a focus on the mentioned aspects. For a more comprehensive response, additional information from the document or a more specific question may help clarify the subject matter."

            # Update content area
            st.markdown(f"""
<div id="contentArea" class="mt-4 p-4 bg-light rounded shadow">
  <div class="alert alert-info" role="alert">
    <h5 class="alert-heading">Answer:</h5>
    <p>{result}</p>
  </div>
</div>
""", unsafe_allow_html=True)

            # Store in chat history
            st.session_state.chat_history.append({"question": question, "answer": result})

        except Exception as e:
            st.markdown(f'<div id="contentArea" class="mt-4 p-4 bg-light rounded shadow"><p class="text-danger">Error generating answer: {str(e)}</p></div>', unsafe_allow_html=True)

# Display chat history
if st.session_state.chat_history:
    st.markdown('<h3 class="mt-5">Chat History</h3>', unsafe_allow_html=True)
    for entry in reversed(st.session_state.chat_history[-3:]):
        st.markdown(f"""
<div class="alert alert-secondary">
  <strong>Q:</strong> {entry['question']}<br>
  <strong>A:</strong> {entry['answer']}
</div>
""", unsafe_allow_html=True)

# About section
st.markdown("""
<div id="about" class="about-container mt-5">
  <div class="about-content">
    <div class="about-text">
      <div>Welcome to Quick Veda: Your Document Analysis Solution</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar with tips
with st.sidebar:
    st.markdown("""
### Tips for Better Results
- Upload a clear PDF document.
- Ask specific questions (e.g., "What is the main topic?" instead of "Tell me about it").
- Use questions starting with "Explain," "Describe," or "Summarize" for detailed answers.
- Processing may take a moment for large files.
    """)
    st.markdown("""
### About Quick Veda
This app uses Flan-T5 to analyze PDFs and provide detailed answers to your questions. It runs locally, ensuring privacy without API keys.
    """)

# Bootstrap JS
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.8/dist/umd/popper.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.min.js"></script>
""", unsafe_allow_html=True)
