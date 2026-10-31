
# Heritage-RAG 🏛️

## About the Project

Heritage-RAG is an AI-powered application built using Streamlit. It allows users to interact with uploaded PDF documents and ask questions based on their content.

It also provides a **General Knowledge** mode, where users can ask questions without uploading a PDF.

## Features

- Upload and process PDF documents.
- Extract text from PDF files.
- Split documents into smaller text chunks.
- Generate embeddings using Sentence Transformers.
- Store document embeddings in ChromaDB.
- Retrieve relevant information using semantic search.
- Generate answers based on uploaded PDF documents.
- Ask general questions using AI.
- Display source references for PDF-based answers.
- Simple and interactive Streamlit interface.

## Technologies Used

- Python
- Streamlit
- PyMuPDF
- ChromaDB
- Sentence Transformers
- Hugging Face Transformers
- Qwen2.5-0.5B-Instruct

## AI Models Used

**Embedding Model:**
`sentence-transformers/all-MiniLM-L6-v2`

**Language Model:**
`Qwen/Qwen2.5-0.5B-Instruct`

## Project Structure

```text
Heritage-RAG/
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── heritage_rag_data/
    └── chroma_db/
```

The ChromaDB folder is created locally when the application runs.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/shirisha4940/Heritage-RAG-new-1

### 2. Open the Project Folder

```bash
cd heritage-rag
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run streamlit_app.py
```

## How to Use

1. Open the Heritage-RAG application.
2. Select **General Knowledge** to ask general questions.
3. Select **Uploaded PDFs** to work with PDF documents.
4. Upload a PDF document.
5. Ask questions related to the uploaded document.
6. View the generated answers and source references.

## Deployment

The application is deployed using Streamlit Community Cloud and connected to a GitHub repository.

## Future Enhancements

- Support for multiple document formats.
- Improved answer accuracy.
- Multilingual document support.
- Better document summarization.
- Improved user interface.

## Conclusion

Heritage-RAG combines Retrieval-Augmented Generation (RAG) with general AI question answering to provide an interactive platform for exploring documents and asking questions.
