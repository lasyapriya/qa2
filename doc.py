import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
import tempfile
import os

# Streamlit page configuration
st.set_page_config(page_title="Quick Veda | Doc Analyzer", page_icon="📄", layout="wide")

# Cache RoBERTa pipeline
@st.cache_resource
def load_qa_pipeline():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2"
    )

# Initialize session state
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# HTML and CSS (unchanged)
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
    body { margin: 0; }
    html { -ms-overflow-style: none; scrollbar-width: none; }
    html::-webkit-scrollbar { display: none; }
    .navbar-brand img { width: 32px; height: 32px; border-radius: 50%; }
    .navbar-brand span { font-size: larger; font-family: fantasy; padding-left: 10px; }
    .logo { height: 40px; }
    li { list-style: none; display: flex; justify-content: space-around; }
    .card-img-overlay { flex-direction: column; display: flex; align-items: center; justify-content: center; }
    .card-img { height: 500px; object-fit: cover; }
    .btn.about { color: burlywood; margin: 10px; font-size: 1.5rem; border-radius: 10px; background-color: #FFFFFF; }
    .content { width: 100vw; }
    .input { display: flex; align-items: center; justify-content: space-between; margin-top: 20px; }
    .upload-label { font-size: 24px; cursor: pointer; }
    .upload-btn { background-color: #007bff; color: white; border: none; padding: 10px 20px; font-size: 16px; cursor: pointer; border-radius: 5px; }
    .upload-btn:hover { background-color: #0056b3; }
    #uploadStatus { margin-top: 20px; font-size: 18px; }
    #contentArea { margin-top: 30px; }
    #paper { transform: rotate(-44deg); }
    .about-container { background-image: url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e'); background-size: cover; background-position: center; height: 50vh; display: flex; align-items: center; justify-content: center; }
    .about-content { display: flex; flex-direction: row; justify-content: center; align-items: center; }
    .about-text { border-radius: 10px; background-color: #FFFFFF; width: 80vw; height: 30vh; box-shadow: 0 0 10px #007bff; flex: 1; display: flex; justify-content: center; align-items: center; text-align: center; }
    .about-text div { font-size: 24px; font-weight: bold; color: rgb(223, 143, 45); }
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
<div id="contentArea" class="mt-4 p-4 bg-light rounded shadow" style="min-height: 200px;">
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

        # Extract text from PDF
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            # Limit to 300 words for better context
            st.session_state.pdf_text = " ".join(text.split()[:300])

            # Clean up
            os.unlink(pdf_path)
        except Exception as e:
            st.markdown(f'<div id="contentArea" class="mt-4 p-4 bg-light rounded shadow"><p class="text-danger">Failed to process PDF: {str(e)}</p></div>', unsafe_allow_html=True)
            st.stop()

    with st.spinner("Generating answer..."):
        try:
            # Run RoBERTa pipeline
            qa_pipeline = load_qa_pipeline()
            result = qa_pipeline(question=question, context=st.session_state.pdf_text)
            answer = result["answer"]

            # Ensure concise output (2–4 lines, 20–40 words)
            if len(answer.split()) > 40:
                answer = " ".join(answer.split()[:40]) + "..."

            # Update content area
            st.markdown(f"""
<div id="contentArea" class="mt-4 p-4 bg-light rounded shadow">
  <div class="alert alert-info" role="alert">
    <h5 class="alert-heading">Answer:</h5>
    <p>{answer}</p>
  </div>
</div>
""", unsafe_allow_html=True)

            # Store in chat history
            st.session_state.chat_history.append({"question": question, "answer": answer})

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
- Ask specific questions (e.g., "What is the main cause of climate change?").
- Expect 2–4 line answers in under 20 seconds.
    """)
    st.markdown("""
### About Quick Veda
This app uses RoBERTa to quickly analyze PDFs and provide concise, relevant answers. No API keys needed.
    """)

# Bootstrap JS
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.8/dist/umd/popper.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.min.js"></script>
""", unsafe_allow_html=True)
